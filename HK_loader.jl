# module HKLoader

using CSV, DataFrames, Dates, JSON3, Glob, Random

# === Konfiguration (spiegelt dein Setup) ======================================
const ORDERED_BASE = [
    "wind_mean_ms","wind_max_ms","wind_min_ms",
    "rpm_mean","rpm_max","rpm_min",
    "power_mean_kw","power_max_kw","power_min_kw",
    "power_avail_wind_mean_kw",
    "power_avail_tech_mean_kw",
    "power_avail_force_maj_mean_kw",
    "power_avail_ext_mean_kw",
]
const ORDERED_BASE_SYMBOLS = Symbol.(ORDERED_BASE)
# const D_OUT = length(ORDERED_BASE)         # 13
# const D_IN  = 4 + 2 + D_OUT                # 19
# const SLOTS = 144                          # 10-Minuten-Raster/Tag

# # === Zeit-LUT (driftfrei) =====================================================
# const Δθ = 2f0 * π / Float32(SLOTS)
# const TIME_LUT = let M = Matrix{Float32}(undef, 2, SLOTS)
#     @inbounds for k in 1:SLOTS
#         s, c = sincos(Float32(k-1) * Δθ)   # Base.sincos ist effizient
#         M[1,k] = s; M[2,k] = c
#     end
#     M
# end

@inline function floor_slot_index(dt::DateTime)::Int
    # „14:21“ → Slot für 14:20 (floor auf 10-Minuten-Raster)
    moday = hour(dt)*60 + minute(dt)              # 0..1439
    slot  = fld(moday, 10) + 1                    # 1..144
    return slot
end

# === Season-Logik (String & OneHot) ===========================================
@inline function season_str(dt::DateTime)::String
    m = month(dt)
    return (m in (12,1,2)) ? "Winter" :
           (m in (3,4,5))  ? "Fruehling" :
           (m in (6,7,8))  ? "Sommer" : "Herbst"
end

@inline function season_onehot(s::AbstractString)::NTuple{4,Float32}
    # Reihenfolge: Winter, Fruehling, Sommer, Herbst
    return (Float32(s=="Winter"),
            Float32(s=="Fruehling"),
            Float32(s=="Sommer"),
            Float32(s=="Herbst"))
end

# === Stats laden: mu/sigma nach (season, serial, feature) =====================
"""
    load_stats(json_path) -> Dict{String, Dict{String, Dict{String, NamedTuple}}}

Gibt `stats[s][serial][feat] => (mu=Float64, sigma=Float64)` zurück.
Fehlende Einträge kannst du später mit Fallbacks behandeln.
"""
function load_stats(json_path::AbstractString)
    raw = JSON3.read(read(json_path, String))
    stats = Dict{String, Dict{String, Dict{String, NamedTuple{(:mu,:sigma),Tuple{Float64,Float64}}}}}()
    for (skey, serials) in pairs(raw)
        sdict = Dict{String, Dict{String, NamedTuple{(:mu,:sigma),Tuple{Float64,Float64}}}}()
        for (serial, feats) in pairs(serials)
            fdict = Dict{String, NamedTuple{(:mu,:sigma),Tuple{Float64,Float64}}}()
            for (feat, obj) in pairs(feats)
                mu    = try obj["mu"]   catch; obj.mu   end
                sigma = try obj["sigma"] catch; obj.sigma end
                fdict[String(feat)] = (mu=Float64(mu), sigma=Float64(sigma))
            end
            sdict[String(serial)] = fdict
        end
        stats[String(skey)] = sdict
    end
    return stats
end


"""
    load_stats_year(json_path) -> Dict{String, Dict{String, NamedTuple}}

Liest `serial -> feat -> {mu, sigma, n}` und gibt
`stats_year[serial][feat] => (mu::Float64, sigma::Float64)` zurück.
"""
function load_stats_year(json_path::AbstractString = joinpath(base_dir, "stats_by_serial_year.json"))
    raw = JSON3.read(read(json_path, String))
    stats = Dict{String, Dict{String, NamedTuple{(:mu,:sigma),Tuple{Float64,Float64}}}}()
    for (serial, feats) in pairs(raw)
        fdict = Dict{String, NamedTuple{(:mu,:sigma),Tuple{Float64,Float64}}}()
        for (feat, obj) in pairs(feats)
            mu    = try obj["mu"]    catch; obj.mu    end
            sigma = try obj["sigma"] catch; obj.sigma end
            fdict[String(feat)] = (mu=Float64(mu), sigma=Float64(sigma))
        end
        stats[String(serial)] = fdict
    end
    return stats
end

# Jahr-Fallback: (mu=0, sigma=1), wenn kein Eintrag existiert
get_stats_year(stats_year, serial::String, feat::String) =
    haskey(stats_year, serial) && haskey(stats_year[serial], feat) ?
        stats_year[serial][feat] : (mu=0.0, sigma=1.0)


"""
    load_stats_flexible(json_path) -> (:season, stats) oder (:year, stats_year)

Erkennt am Top-Level, ob Seasons oder Serials stehen.
"""
function load_stats_flexible(json_path::AbstractString)
    raw = JSON3.read(read(json_path, String))
    keys_top = Set(String.(collect(keys(raw))))
    seasons = Set(["Winter","Fruehling","Sommer","Herbst"])
    if !isempty(intersect(keys_top, seasons))  # Season-Form
        return (:season, load_stats(json_path))
    else                                       # Year-Form (serial -> feat)
        return (:year, load_stats_year(json_path))
    end
end


# --- Log1p-Normalizer ---------------------------------------------------------
struct Log1pNorm
    scale::Vector{Float32}   # s_j > 0, Länge D_OUT
    mu_z::Vector{Float32}    # Mittel im log1p-Raum, Länge D_OUT
    sig_z::Vector{Float32}   # Std im log1p-Raum, Länge D_OUT
end

const LOG1P_CACHE = Dict{NTuple{4,String}, Log1pNorm}()
const LOG1P_CACHE_LOADED = Set{String}()

log1p_cache_key(base_dir::AbstractString, serial::AbstractString,
                seasons, stats_path_year::AbstractString) =
    (abspath(base_dir),
     String(serial),
     join(String.(collect(seasons)), "|"),
     abspath(stats_path_year))

log1p_cache_file(base_dir::AbstractString) =
    joinpath(abspath(base_dir), "stats_log1p_cache.json")

@inline serialize_log1p_key(key::NTuple{4,String}) = join(key, "||")

function deserialize_log1p_key(keystr::AbstractString)
    parts = split(keystr, "||")
    length(parts) == 4 || error("Invalid log1p cache key: $keystr")
    return (String(parts[1]), String(parts[2]), String(parts[3]), String(parts[4]))
end

function ensure_log1p_cache_loaded!(base_dir::AbstractString)
    abs_base = abspath(base_dir)
    abs_base in LOG1P_CACHE_LOADED && return
    path = log1p_cache_file(base_dir)
    if isfile(path)
        raw = JSON3.read(read(path, String))
        for (ks, obj) in pairs(raw)
            key = deserialize_log1p_key(String(ks))
            scale = Float32.(obj["scale"])
            mu_z  = Float32.(obj["mu_z"])
            sig_z = Float32.(obj["sig_z"])
            LOG1P_CACHE[key] = Log1pNorm(collect(scale), collect(mu_z), collect(sig_z))
        end
    end
    push!(LOG1P_CACHE_LOADED, abs_base)
end

function persist_log1p_cache!(base_dir::AbstractString)
    abs_base = abspath(base_dir)
    data = Dict{String, Dict{String, Vector{Float64}}}()
    for (key, ln) in LOG1P_CACHE
        key[1] == abs_base || continue
        data[serialize_log1p_key(key)] = Dict(
            "scale" => collect(Float64.(ln.scale)),
            "mu_z"  => collect(Float64.(ln.mu_z)),
            "sig_z" => collect(Float64.(ln.sig_z)),
        )
    end
    path = log1p_cache_file(base_dir)
    try
        open(path, "w") do io
            JSON3.pretty(io, JSON3.write(data))
        end
    catch err
        @warn "Failed to persist log1p cache" path err
    end
end

function gather_block_entries(base_dir::AbstractString, serial::AbstractString, seasons)
    entries = Tuple{String,Vector{String}}[]
    for s in seasons
        dir_s = joinpath(base_dir, s, serial)
        isdir(dir_s) || continue
        files = sort(glob("*.csv", dir_s))
        isempty(files) && continue
        push!(entries, (String(s), files))
    end
    return entries
end

function build_log1p_scale(stats_year::Dict{String,<:Any}, serial::String)
    scale = Vector{Float32}(undef, D_OUT)
    @inbounds for (i, feat) in enumerate(ORDERED_BASE)
        st = get_stats_year(stats_year, serial, feat)
        μ = Float32(max(st.mu, 0.0))
        σ = Float32(max(st.sigma, 1e-6))
        candidate = μ + 3f0*σ
        scale[i] = candidate > 1f-3 ? candidate : 1f0
    end
    return scale
end

function compute_log1p_stats_from_files(files::Vector{String}, scale::Vector{Float32})
    μ = zeros(Float64, D_OUT)
    m2 = zeros(Float64, D_OUT)
    counts = zeros(Int, D_OUT)
    scale64 = Float64.(scale)
    for file in files
        df = CSV.read(file, DataFrame; normalizenames=false)
        for row in eachrow(df)
            @inbounds for (j, sym) in enumerate(ORDERED_BASE_SYMBOLS)
                raw = row[sym]
                if raw === missing
                    continue
                end
                val = Float64(raw)
                if !isfinite(val)
                    continue
                end
                val = max(val, 0.0)
                z = log1p(val / scale64[j])
                counts[j] += 1
                δ = z - μ[j]
                μ[j] += δ / counts[j]
                m2[j] += δ * (z - μ[j])
            end
        end
    end
    mu_z = Vector{Float32}(undef, D_OUT)
    sig_z = Vector{Float32}(undef, D_OUT)
    @inbounds for j in 1:D_OUT
        mu_z[j] = Float32(counts[j] == 0 ? 0.0 : μ[j])
        if counts[j] > 1
            σ = sqrt(m2[j] / (counts[j] - 1))
            sig_z[j] = σ > 1f-6 ? Float32(σ) : 1f0
        else
            sig_z[j] = 1f0
        end
    end
    return mu_z, sig_z
end

function build_log1p_norm(base_dir::AbstractString, serial::AbstractString;
                          seasons = ("Winter","Fruehling","Sommer","Herbst"),
                          stats_year::Union{Nothing,Dict}=nothing,
                          stats_path_year::AbstractString = joinpath(base_dir, "stats_by_serial_year.json"),
                          block_entries::Union{Nothing,Vector{Tuple{String,Vector{String}}}}=nothing)
    ensure_log1p_cache_loaded!(base_dir)
    key = log1p_cache_key(base_dir, serial, seasons, stats_path_year)
    if haskey(LOG1P_CACHE, key)
        return LOG1P_CACHE[key]
    end
    stats = isnothing(stats_year) ? load_stats_year(stats_path_year) : stats_year
    entries = isnothing(block_entries) ? gather_block_entries(base_dir, serial, seasons) : block_entries
    files = String[]
    for (_, flist) in entries
        append!(files, flist)
    end
    isempty(files) && error("No CSV files found for serial=$serial under $base_dir")
    scale = build_log1p_scale(stats, String(serial))
    mu_z, sig_z = compute_log1p_stats_from_files(files, scale)
    ln = Log1pNorm(scale, mu_z, sig_z)
    LOG1P_CACHE[key] = ln
    persist_log1p_cache!(base_dir)
    return ln
end

# vorwärts/invers für einen (D_OUT, T)-Block
@inline function fwd_log1p_zscore!(Y::AbstractMatrix{Float32}, ln::Log1pNorm)
    @views Y .= (log1p.(Y ./ ln.scale) .- ln.mu_z) ./ ln.sig_z
    return Y
end

@inline function inv_log1p_zscore!(Y::AbstractMatrix{Float32}, ln::Log1pNorm)
    @views Y .= ln.scale .* (expm1.(Y .* ln.sig_z .+ ln.mu_z))
    Y .= max.(Y, 0f0)   # nur für numerische Sicherheit
    return Y
end




# Hilfs-Fallback: wenn stats fehlen → (mu=0, sigma=1) oder ggf. saisonweiter Mittelwert
function get_stats(stats, s::String, serial::String, feat::String)
    if haskey(stats, s) && haskey(stats[s], serial) && haskey(stats[s][serial], feat)
        return stats[s][serial][feat]
    elseif haskey(stats, s)
        # saisonweiter Mittelwert als Fallback, wenn vorhanden
        acc_mu = 0.0; acc_s = 0.0; cnt = 0
        for (ser, fdict) in stats[s]
            if haskey(fdict, feat)
                acc_mu += fdict[feat].mu
                acc_s  += fdict[feat].sigma
                cnt += 1
            end
        end
        if cnt > 0
            return (mu = acc_mu/cnt, sigma = max(acc_s/cnt, 1e-6))
        end
    end
    return (mu = 0.0, sigma = 1.0)
end



mutable struct Agg
    n::Int
    mean::Float64
    m2::Float64
end
Agg() = Agg(0, 0.0, 0.0)

# Merge (μ_i, σ_i, n_i) in einen Aggregator (Chan/Golub/LeVeque)
function update!(a::Agg, μ::Float64, σ::Float64, n::Int)
    n <= 0 && return a
    if a.n == 0
        a.n = n
        a.mean = μ
        a.m2 = (σ^2) * (n - 1)
        return a
    end
    N, M, M2 = a.n, a.mean, a.m2
    δ   = μ - M
    Np  = N + n
    Mp  = M + δ * n / Np
    M2p = M2 + (σ^2) * (n - 1) + δ^2 * (N * n / Np)
    a.n, a.mean, a.m2 = Np, Mp, M2p
    return a
end

finalize(a::Agg) = a.n <= 1 ? (mu=a.mean, sigma=0.0, n=a.n) : (mu=a.mean, sigma=sqrt(a.m2/(a.n-1)), n=a.n)

function aggregate_year_stats(in_json::AbstractString, out_json::AbstractString="stats_by_serial_year.json")
    raw = JSON3.read(read(in_json, String))  # season -> serial -> feat -> {mu,sigma,n}
    # acc[serial][feat] = Agg()
    acc = Dict{String, Dict{String, Agg}}()
    for (season, serials) in pairs(raw)
        for (serial, feats) in pairs(serials)
            sdict = get!(acc, String(serial), Dict{String, Agg}())
            for (feat, obj) in pairs(feats)
                μ = try obj["mu"]    catch; obj.mu    end
                σ = try obj["sigma"] catch; obj.sigma end
                n = try obj["n"]     catch; 0         end
                update!(get!(sdict, String(feat), Agg()), Float64(μ), Float64(σ), Int(n))
            end
        end
    end
    # finalisieren
    out = Dict{String, Dict{String, Dict{String, Float64}}}()
    for (serial, feats) in acc
        yd = Dict{String, Dict{String, Float64}}()
        for (feat, a) in feats
            st = finalize(a)
            yd[feat] = Dict("mu"=>st.mu, "sigma"=>st.sigma, "n"=>float(st.n))
        end
        out[serial] = yd
    end
    open(out_json, "w") do io
        #JSON3.write(io, out; allow_inf=true, indent=2)
        JSON3.pretty(io, JSON3.write(out))
    end
    println("Wrote ", out_json)
end





# === Zeile → 19-D-Embedding (OneHot Season, Zeit aus LUT, 13 normierte Kanäle)
"""
    embed_row(row, serial, stats) -> Vector{Float32}

- liest `row[:time]` (String oder DateTime),
- bestimmt Season & Slot (floor auf 10 Minuten),
- normalisiert ORDERED_BASE mit stats[(season, serial, feat)].
"""
function embed_row(row, serial::String, stats;
                   norm::Symbol = :year,
                   lognorm::Union{Nothing,Log1pNorm}=nothing)
    # Zeit/Season
    dt = row[:time] isa DateTime ? row[:time] : DateTime(row[:time])
    s  = season_str(dt)
    s4 = season_onehot(s)
    k  = floor_slot_index(dt)

    # Numerik: z-Score per Season ODER per Year (pro Serial)
    z = Vector{Float32}(undef, D_OUT)
    @inbounds for (i, feat) in enumerate(ORDERED_BASE)
        v = row[Symbol(feat)]
        if norm === :year_log1p
            isnothing(lognorm) && error("lognorm params missing for :year_log1p mode")
            val = v === missing ? 0.0f0 : Float32(v)
            val = ifelse(isfinite(val), val, 0.0f0)
            val = max(val, 0f0)
            s_j  = lognorm.scale[i]
            μz   = lognorm.mu_z[i]
            σz   = max(lognorm.sig_z[i], 1f-6)
            z[i] = (log1p(val / s_j) - μz) / σz
        else
            v = (v === missing) ? NaN : v
            ms = norm === :season ? get_stats(stats, s, serial, feat) :
                 norm === :year   ? get_stats_year(stats, serial, feat) :
                 error("Unknown norm mode: $norm")
            σ  = max(ms.sigma, 1e-6)
            z[i] = Float32((Float64(v) - ms.mu) / σ)
        end
    end

    emb = Vector{Float32}(undef, D_IN)
    emb[1:4]   .= (s4[1], s4[2], s4[3], s4[4])
    emb[5]     = TIME_LUT[1, k]
    emb[6]     = TIME_LUT[2, k]
    emb[7:end] .= z
    return emb
end

# === CSV-Blöcke einer Seriennummer laden =====================================
"""
    load_blocks_for_serial(base_dir, serial; seasons=("Winter","Fruehling","Sommer","Herbst"), stats_path)

Liest alle CSVs unter `base_dir/Season/serial/*.csv`, baut pro CSV ein
`Matrix{Float32} (D_IN, T)` mit Embeddings.
"""
function load_blocks_for_serial(base_dir::AbstractString, serial::AbstractString;
                                seasons = ("Winter","Fruehling","Sommer","Herbst"),
                                stats_path::AbstractString = joinpath(base_dir, "stats_by_season_serial.json"),
                                norm::Symbol = :year_log1p,
                                stats_path_year::AbstractString = joinpath(base_dir, "stats_by_serial_year.json"))
    serial_str = String(serial)
    block_entries = gather_block_entries(base_dir, serial_str, seasons)

    global seqs
    seqs = Vector{Matrix{Float32}}()
    isempty(block_entries) && return seqs

    stats = nothing
    stats_year = nothing
    lognorm = nothing

    if norm === :season
        stats = load_stats(stats_path)
    elseif norm === :year
        stats_year = load_stats_year(stats_path_year)
        stats = stats_year
    elseif norm === :year_log1p
        stats_year = load_stats_year(stats_path_year)
        stats = stats_year
        lognorm = build_log1p_norm(base_dir, serial_str;
                                   seasons=seasons,
                                   stats_year=stats_year,
                                   stats_path_year=stats_path_year,
                                   block_entries=block_entries)
    else
        error("Unknown norm mode: $norm")
    end

    for (_, files) in block_entries
        for f in files
            df = CSV.read(f, DataFrame; normalizenames=false)
            @assert "time" ∈ names(df) "Spalte 'time' fehlt in $f"
            T = nrow(df)
            E = Matrix{Float32}(undef, D_IN, T)
            @inbounds for t in 1:T
                row = df[t, :]
                E[:, t] = embed_row(row, serial_str, stats;
                                    norm=norm,
                                    lognorm=lognorm)
            end
            push!(seqs, E)
        end
    end
    return seqs
end

# === Fenster + Minibatches ====================================================
"""
    make_window_index(seqs, ctx) -> Vector{Tuple{Int,Int}}

Erzeugt Indizes (seq_id, t) für alle t mit t ≥ ctx und t < T (weil Ziel y_{t+1}).
"""
function make_window_index(seqs::Vector{<:AbstractMatrix}, ctx::Int)
    idx = Vector{Tuple{Int,Int}}()
    for (i, E) in enumerate(seqs)
        T = size(E, 2)
        for t in ctx:(T-1)
            push!(idx, (i, t))
        end
    end
    return idx
end

"""
    next_batch!(X, Y, seqs, idx, p, ctx)

Füllt Batch-Matrizen:
- X :: (D_IN, ctx, B)
- Y :: (D_OUT, B)  (Ziel: numerische Kanäle der Zeile t+1, also Embedding[7:19])
ab Position p in idx.
"""
function next_batch!(X::AbstractArray, Y::AbstractArray,
                     seqs::Vector{<:AbstractMatrix},
                     idx::Vector{Tuple{Int,Int}}, p::Int, ctx::Int)
    B = size(Y, 3)
    @inbounds for b in 1:B
        (si, t) = idx[p + b - 1]
        E = seqs[si]
        X[:, :, b] = @view E[:, t-ctx+1 : t]          # (D_IN, ctx)
        Y[:, :, b]    = @view E[7:6+D_OUT, t-ctx+2 : t+1] # (D_OUT, ctx)
    end
end

"""
    struct HKWindowLoader

Custom-Loader, der (X,Y)-Minibatches über alle Fenster liefert.
"""
struct HKWindowLoader
    seqs::Vector{Matrix{Float32}}
    idx::Vector{Tuple{Int,Int}}
    ctx::Int
    batchsize::Int
end

Base.length(dl::HKWindowLoader) = cld(length(dl.idx), dl.batchsize)

function Base.iterate(dl::HKWindowLoader, state::Int=1)
    N = length(dl.idx)
    state > N && return nothing
    last = min(state + dl.batchsize - 1, N)
    B    = last - state + 1
    X = Array{Float32}(undef, D_IN, dl.ctx, B)
    Y = Array{Float32}(undef, D_OUT, dl.ctx, B)
    next_batch!(X, Y, dl.seqs, dl.idx, state, dl.ctx)
    return ((X, Y), last + 1)
end

"""
    make_loader(seqs; ctx=20, batchsize=32, shuffle=true, rng=Random.GLOBAL_RNG)

Erstellt einen HKWindowLoader. Shuffle mischt die Fenster.
"""
function make_loader(seqs::Vector{Matrix{Float32}}; ctx::Int=20, batchsize::Int=32,
                     shuffle::Bool=true, rng=Random.GLOBAL_RNG)
    idx = make_window_index(seqs, ctx)
    shuffle && Random.shuffle!(rng, idx)
    return HKWindowLoader(seqs, idx, ctx, batchsize)
end

# end # module



function data_check(X)
    for i in 1:size(X,3)
        m = X[:,:,i]
        if minimum(m) < -10 || maximum(m) > 10 || isnan(minimum(m))
            @show i
            @show m
        end
    end
end

function seqs_check()
    for i in 1:length(seqs)
        m = seqs[i]
        if minimum(m) < -10 || maximum(m) > 10 || isnan(minimum(m))
            @show i
            @show m
        end
    end
end
