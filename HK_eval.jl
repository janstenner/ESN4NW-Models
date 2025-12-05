

using Random, LinearAlgebra
using Dates, CSV, DataFrames
using PlotlyJS            # pkg> add PlotlyJS


const SEASON_NAMES = ("Winter","Fruehling","Sommer","Herbst")

const PLOTLY13 = [
    # wind_*  (Blues)
    "#3182BD",  # wind_mean_ms  (mittelblau)
    "#08519C",  # wind_max_ms   (dunkelblau)
    "#9ECAE1",  # wind_min_ms   (hellblau)

    # rpm_*   (Oranges)
    "#E6550D",  # rpm_mean
    "#A63603",  # rpm_max
    "#FDAE6B",  # rpm_min

    # power_* (Greens)
    "#31A354",  # power_mean_kw
    "#006D2C",  # power_max_kw
    "#74C476",  # power_min_kw

    # power_avail_* (Purples)
    "#54278F",  # power_avail_wind_mean_kw
    "#756BB1",  # power_avail_tech_mean_kw
    "#9E9AC8",  # power_avail_force_maj_mean_kw
    "#B6B6EAFF",  # power_avail_ext_mean_kw
]

"""
    RunResult

Container für einen 500er-Run (real vs. pred) aus einem Block.
- `season`  : String ("Winter" | "Fruehling" | "Sommer" | "Herbst")
- `seq_id`  : Index im `seqs`-Vektor
- `start`   : Startindex (1-basiert) innerhalb der Sequenz
- `real`    : (D_OUT, L)   Zielsoll, L=run_len
- `pred`    : (D_OUT, L)   autoregressiv generiert
"""
mutable struct RunResult
    season::String
    seq_id::Int
    start::Int
    real::Matrix{Float32}
    pred::Matrix{Float32}
end

# --- Hilfsfunktionen ----------------------------------------------------------

@inline function season_from_onehot(v::AbstractVector{<:Real})::String
    @assert length(v) == 4
    i = findmax(v)[2]
    return SEASON_NAMES[i]
end

@inline function nearest_slot_from_sc(sinval::Real, cosval::Real)::Int
    bestk = 1
    bestd = typemax(Float32)
    @inbounds for k in 1:SLOTS
        ds = Float32(TIME_LUT[1,k]) - Float32(sinval)
        dc = Float32(TIME_LUT[2,k]) - Float32(cosval)
        d  = ds*ds + dc*dc
        if d < bestd
            bestd = d; bestk = k
        end
    end
    return bestk
end

# Temperaturiertes Faktor-Sampling: y = μ + sqrt(τ)*(U z1 + σ ⊙ z2)
function sample_factor_tau(mu::AbstractVector{<:Real},
                           logσ::AbstractVector{<:Real},
                           U::AbstractMatrix{<:Real}; τ::Float32=1f0)
    D = length(mu); r = size(U,2)
    σ  = exp.(Float32.(logσ)) .+ 1f-6
    z1 = randn(Float32, r)
    z2 = randn(Float32, D)
    return Float32.(mu) .+ sqrt(τ) .* (Float32.(U) * z1 .+ σ .* z2)
end

# Nimmt eine (D_IN, T)-Sequenz E, wählt random Start, erzeugt real & pred (beide Länge L)
function generate_run!(model::ARTransformer, E::AbstractMatrix{<:Real};
                       run_len::Int=500, ctx::Int=20, τ::Float32=1f0)
    T = size(E, 2)
    @assert T >= ctx + run_len "Sequenz zu kurz: T=$T < ctx+run_len=$(ctx+run_len)"
    s = rand(1:(T - (ctx + run_len) + 1))
    # Seed-Kontext X
    X = reshape(Float32.(E[:, s:s+ctx-1]), D_IN, ctx, 1)

    # initialer Slot-Index aus letzter Kontextspalte
    s0 = X[5, end, 1]; c0 = X[6, end, 1]
    t_idx = nearest_slot_from_sc(s0, c0)

    # Ground-truth für Vergleich (Zielkanäle sind Embedding[7:6+D_OUT, ...])
    real = Array{Float32}(undef, D_OUT, run_len)
    @inbounds real[:, :] = Float32.(E[7:6+D_OUT, s+ctx : s+ctx+run_len-1])

    # Autoregressiv generieren
    pred = Array{Float32}(undef, D_OUT, run_len)
    @inbounds for i in 1:run_len
        μ, logσ, U = model(X)                            # (D_OUT,T,B) etc.
        μt    = vec(μ[:, end, 1])
        logσt = vec(logσ[:, end, 1])
        Ut    = Array(U[:, :, end, 1])                   # (D_OUT, RANK)

        ŷ = sample_factor_tau(μt, logσt, Ut; τ=τ)        # (D_OUT,)
        pred[:, i] = ŷ

        xin = X[:, end, 1]
        xin_next, t_idx = build_input_from_prediction(xin, ŷ, t_idx)
        xin_next3 = reshape(xin_next, D_IN, 1, 1)

        # Sliding Window (Kontext beibehalten)
        if size(X, 2) < ctx
            X = hcat(X, xin_next3)
        else
            X = cat(@view(X[:, 2:end, :]), xin_next3; dims=2)
        end
    end

    season = season_from_onehot(@view E[1:4, 1])
    return RunResult(season, 0, s, real, pred)  # seq_id wird später gesetzt
end



# ---- Ganzjahres-Generation ---------------------------------------------------
const SLOT_MINUTES = Int(div(24*60, SLOTS))  # 10-Minuten-Raster pro Slot

"""
    build_year_slot_timeline(year) -> (times, seasons_for_slot)

Erzeugt Vektoren aller 10-Minuten-Slots eines Kalenderjahres (chronologisch),
inkl. der zugehörigen Season-Labels laut `season_str`.
"""
function build_year_slot_timeline(year::Int)
    slot_step = Minute(SLOT_MINUTES)
    times = DateTime[]
    seasons_for_slot = String[]

    for d in Date(year, 1, 1):Day(1):Date(year, 12, 31)
        s = season_str(DateTime(d))
        base_dt = DateTime(d)
        @inbounds for k in 0:(SLOTS-1)
            push!(times, base_dt + slot_step * k)
            push!(seasons_for_slot, s)
        end
    end
    return times, seasons_for_slot
end

@inline function season_onehot_vec(s::AbstractString)
    oh = season_onehot(s)
    return Float32.(collect(oh))
end

function pick_random_context(seqs::Vector{<:AbstractMatrix}, season::AbstractString; ctx::Int=CTX)
    idxs = filter_by_season(seqs, season)
    @assert !isempty(idxs) "Keine Blöcke für Season=$season gefunden."
    for _ in 1:200
        si = rand(idxs)
        E = seqs[si]
        T = size(E, 2)
        T >= ctx && begin
            s = rand(1:(T-ctx+1))
            return Array{Float32}(E[:, s:s+ctx-1])
        end
    end
    error("Finde keinen ausreichend langen Kontext (ctx=$ctx) für Season=$season.")
end

function reset_context_time!(ctx_mat::AbstractMatrix{<:Real})
    ctx = size(ctx_mat, 2)
    start_slot = max(1, SLOTS - ctx + 1)
    slot = start_slot
    @inbounds for col in 1:ctx
        ctx_mat[5, col] = TIME_LUT[1, slot]
        ctx_mat[6, col] = TIME_LUT[2, slot]
        slot += 1
    end
    return ctx_mat
end

function retag_context(ctx_mat::AbstractMatrix{<:Real}, season::AbstractString; reset_time::Bool=true)
    ctx_out = Array{Float32}(ctx_mat)
    oh = season_onehot_vec(season)
    @inbounds for col in 1:size(ctx_out, 2)
        ctx_out[1:4, col] .= oh
    end
    reset_time && reset_context_time!(ctx_out)
    return ctx_out
end

function generate_from_context(model::ARTransformer,
                               ctx_mat::AbstractMatrix{<:Real};
                               run_len::Int,
                               τ::Float32=1f0)
    ctx = size(ctx_mat, 2)
    X = reshape(Float32.(ctx_mat), D_IN, ctx, 1)
    s0 = X[5, end, 1]; c0 = X[6, end, 1]
    t_idx = nearest_slot_from_sc(s0, c0)

    pred = Array{Float32}(undef, D_OUT, run_len)
    real = zeros(Float32, D_OUT, run_len)

    @inbounds for i in 1:run_len
        μ, logσ, U = model(X)
        μt    = vec(μ[:, end, 1])
        logσt = vec(logσ[:, end, 1])
        Ut    = Array(U[:, :, end, 1])

        ŷ = sample_factor_tau(μt, logσt, Ut; τ=τ)
        pred[:, i] = ŷ

        xin = X[:, end, 1]
        xin_next, t_idx = build_input_from_prediction(xin, ŷ, t_idx)
        xin_next3 = reshape(xin_next, D_IN, 1, 1)

        if size(X, 2) < ctx
            X = hcat(X, xin_next3)
        else
            X = cat(@view(X[:, 2:end, :]), xin_next3; dims=2)
        end
    end

    season = season_from_onehot(@view X[1:4, 1, 1])
    rr = RunResult(season, 0, 1, real, pred)
    last_ctx = Array{Float32}(X[:, :, 1])
    return rr, last_ctx
end

function generate_from_context(model::FMTransformer,
                               ctx_mat::AbstractMatrix{<:Real};
                               run_len::Int,
                               τ::Float32=1f0,              # kept for signature parity
                               steps::Int=38,
                               solver::Symbol=:midpoint,
                               seed::Union{Nothing,Int}=nothing)
    ctx = size(ctx_mat, 2)
    X = reshape(Float32.(ctx_mat), D_IN, ctx, 1)
    s0 = X[5, end, 1]; c0 = X[6, end, 1]
    t_idx = nearest_slot_from_sc(s0, c0)

    pred = Array{Float32}(undef, D_OUT, run_len)
    real = zeros(Float32, D_OUT, run_len)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    @inbounds for i in 1:run_len
        t_idx_next = (t_idx == SLOTS) ? 1 : (t_idx + 1)
        x̂ = if solver === :euler
            integrate_cfm_euler(model, X, t_idx_next; steps=steps, rng=rng)
        else
            integrate_cfm_midpoint(model, X, t_idx_next; steps=steps, rng=rng)
        end
        pred[:, i] = x̂

        xin = X[:, end, 1]
        xin_next, t_idx = build_input_from_prediction(xin, x̂, t_idx)
        xin_next3 = reshape(xin_next, D_IN, 1, 1)

        if size(X, 2) < ctx
            X = hcat(X, xin_next3)
        else
            X = cat(@view(X[:, 2:end, :]), xin_next3; dims=2)
        end
    end

    season = season_from_onehot(@view X[1:4, 1, 1])
    rr = RunResult(season, 0, 1, real, pred)
    last_ctx = Array{Float32}(X[:, :, 1])
    return rr, last_ctx
end

function condense_season_segments(seasons_for_slot::Vector{String})
    segments = Tuple{String,Int}[]
    isempty(seasons_for_slot) && return segments
    cur = seasons_for_slot[1]; cnt = 0
    @inbounds for s in seasons_for_slot
        if s == cur
            cnt += 1
        else
            push!(segments, (cur, cnt))
            cur = s; cnt = 1
        end
    end
    push!(segments, (cur, cnt))
    return segments
end

const READABLE_FLOAT_FEATS = Set([
    "wind_mean_ms","wind_max_ms","wind_min_ms",
    "rpm_mean","rpm_max","rpm_min",
])

function apply_value_formatting!(mat::AbstractMatrix)
    @inbounds for (j, feat) in enumerate(ORDERED_BASE)
        if feat in READABLE_FLOAT_FEATS
            mat[j, :] .= round.(Float64.(mat[j, :]); digits=2)
        else
            mat[j, :] .= floor.(max.(Float64.(mat[j, :]), 0.0))
        end
    end
    return mat
end

"""
    generate_year_csv(model; year=2021, serial=SERIAL, base=BASE, ctx=CTX,
                      τ=1f0, norm=:year_log1p, stats_path_year=joinpath(base, "stats_by_serial_year.json"),
                      output_path="generated_year.csv",
                      seasons=SEASON_NAMES, fm_steps=38, fm_solver=:midpoint)

Generiert pro Season genügend 10-Minuten-Schritte, um das Kalenderjahr `year`
vollständig abzudecken, denormalisiert sie und schreibt eine CSV mit
`time` + `ORDERED_BASE`-Spalten.
"""
function generate_year_csv(model;
                           year::Int=2026,
                           serial::AbstractString=SERIAL,
                           base::AbstractString=BASE,
                           ctx::Int=CTX,
                           τ::Float32=1f0,
                           norm::Symbol=:year_log1p,
                           stats_path_year::AbstractString=joinpath(base, "stats_by_serial_year.json"),
                           output_path::AbstractString="generated_year.csv",
                           seasons=SEASON_NAMES,
                           fm_steps::Int=38,
                           fm_solver::Symbol=:midpoint)
    # 1) Zeitachsen + Season-Raster des Zieljahres
    println("→ Baue Zeitachsen für Jahr $year …")
    times, seasons_for_slot = build_year_slot_timeline(year)
    segments = condense_season_segments(seasons_for_slot)
    @inbounds for (s, _) in segments
        @assert s in seasons "Season $s kommt im Jahresraster vor, ist aber nicht in `seasons` erlaubt."
    end
    println("→ Gefundene Segmente (Season, Slots): $(segments)")

    # 2) Daten laden
    println("→ Lade Blöcke und Jahres-Stats …")
    seqs = load_blocks_for_serial(base, serial;
                                  norm=norm,
                                  seasons=seasons,
                                  stats_path_year=stats_path_year)
    stats_year = load_stats_year(stats_path_year)

    # 3) Initialen Kontext aus erster Season ziehen (typisch Winter)
    first_season = first(segments)[1]
    ctx_mat = pick_random_context(seqs, first_season; ctx=ctx)
    println("→ Starte mit Season $first_season, zufälliger Kontextgröße $ctx gewählt.")

    n = length(times)
    year_mat = Array{Float64}(undef, D_OUT, n)

    pos = 0
    last_ctx = ctx_mat

    for (seg_idx, (s, len_slots)) in enumerate(segments)
        println("→ Generiere Segment $seg_idx: $s mit $len_slots Slots …")
        ctx_for_seg = seg_idx == 1 ? last_ctx : retag_context(last_ctx, s; reset_time=true)

        if model isa FMTransformer
            rr, last_ctx = generate_from_context(model, ctx_for_seg;
                                                 run_len=len_slots,
                                                 τ=τ,
                                                 steps=fm_steps,
                                                 solver=fm_solver)
        else
            rr, last_ctx = generate_from_context(model, ctx_for_seg;
                                                 run_len=len_slots,
                                                 τ=τ)
        end

        _, pred_d = denorm_mats(rr, serial, stats_year;
                                cols=ORDERED_BASE,
                                norm=norm,
                                base_dir=base,
                                seasons=seasons,
                                stats_path_year=stats_path_year)
        pred_d64 = Array{Float64}(pred_d)
        apply_value_formatting!(pred_d64)

        @inbounds year_mat[:, pos+1 : pos+len_slots] .= pred_d64[:, 1:len_slots]
        pos += len_slots
    end

    @assert pos == n "Befüllte Slots ($pos) stimmen nicht mit Zeitleiste ($n) überein."

    time_col = [string(t) * ".0" for t in times]
    df = DataFrame()
    df[!, "time"] = time_col
    for (j, feat) in enumerate(ORDERED_BASE)
        col = vec(year_mat[j, :])
        if feat in READABLE_FLOAT_FEATS
            df[!, feat] = col
        else
            df[!, feat] = Int.(col)
        end
    end

    println("→ Schreibe CSV nach $output_path …")
    CSV.write(output_path, df)

    println("→ Plot erstelle …")
    traces_real, traces_pred = traces_for_denorm_mats(year_mat, year_mat; cols=ORDERED_BASE)
    p = plot(vcat(traces_real, traces_pred))
    display(p)

    println("→ Fertig. Gesamt-Slots: $n, Datei: $output_path")
    return output_path
end

"""
    plot_generated_csv(path="generated_year.csv")

Liest eine zuvor erzeugte Jahres-CSV (z.B. via `generate_year_csv`) und plottet
alle Kanäle analog zu `traces_for_denorm_mats`.
"""
function plot_generated_csv(path::AbstractString="generated_year.csv")
    println("→ Lade CSV aus $path …")
    df = CSV.read(path, DataFrame; normalizenames=false)
    @assert "time" in names(df) "Spalte 'time' fehlt in CSV."
    for feat in ORDERED_BASE
        @assert feat in names(df) "Spalte $feat fehlt in CSV."
    end

    n = nrow(df)
    mat = Array{Float64}(undef, D_OUT, n)
    @inbounds for (j, feat) in enumerate(ORDERED_BASE)
        mat[j, :] .= Float64.(df[!, feat])
    end

    println("→ Erzeuge Plot …")
    traces_real, traces_pred = traces_for_denorm_mats(mat, mat; cols=ORDERED_BASE)
    p = plot(vcat(traces_real, traces_pred))
    display(p)
    return p
end

"""
    interpolate_generated_csv(path="generated_year.csv")

Liest eine generierte Jahres-CSV (10-Minuten-Raster), interpoliert auf 15-Minuten-
Raster gemäß der 10/5-Minuten-Gewichte und speichert als `*_15min.csv` mit
der gleichen Rundungslogik wie `generate_year_csv`. Plottet anschließend Original
und interpoliertes Set untereinander.
"""
function interpolate_generated_csv(path::AbstractString="generated_year.csv")
    println("→ Lade CSV aus $path …")
    df = CSV.read(path, DataFrame; normalizenames=false)
    @assert "time" in names(df) "Spalte 'time' fehlt in CSV."
    for feat in ORDERED_BASE
        @assert feat in names(df) "Spalte $feat fehlt in CSV."
    end

    n_orig = nrow(df)
    times_orig = Vector{DateTime}(undef, n_orig)
    @inbounds for i in 1:n_orig
        rawt = df[!, "time"][i]
        tstr = string(rawt)
        tclean = replace(tstr, r"\\.0$" => "")
        times_orig[i] = DateTime(tclean)
    end

    orig_mat = Array{Float64}(undef, D_OUT, n_orig)
    @inbounds for (j, feat) in enumerate(ORDERED_BASE)
        orig_mat[j, :] .= Float64.(df[!, feat])
    end

    start_t = times_orig[1]
    end_t   = times_orig[end]
    step15  = Minute(15)
    times_15 = DateTime[]
    tcur = start_t
    while tcur <= end_t
        push!(times_15, tcur)
        tcur += step15
    end
    n_new = length(times_15)
    println("→ Interpoliere auf 15-Minuten-Raster: $n_orig → $n_new Zeilen …")

    new_mat = Array{Float64}(undef, D_OUT, n_new)
    @inbounds for i in 1:n_new
        if i == 1
            new_mat[:, i] .= orig_mat[:, 1]
            continue
        end
        target = times_15[i]
        minutes = Int(fld(Dates.value(target - start_t), 60_000))
        window_start = minutes - 15
        window_end   = minutes

        idx_high = min(Int(ceil(minutes / 10)) + 1, n_orig)
        idx_low  = max(1, idx_high - 1)

        low_end   = (idx_low  - 1) * 10
        low_start = low_end - 10
        high_end   = (idx_high - 1) * 10
        high_start = high_end - 10

        overlap_low  = max(0, min(window_end, low_end)  - max(window_start, low_start))
        overlap_high = max(0, min(window_end, high_end) - max(window_start, high_start))
        total = overlap_low + overlap_high

        total > 0 || error("Keine Überlappung für Index $i, minutes=$minutes")

        @views new_mat[:, i] .= (overlap_low .* orig_mat[:, idx_low] .+ overlap_high .* orig_mat[:, idx_high]) ./ total
    end

    apply_value_formatting!(new_mat)

    root, ext = splitext(path)
    ext_final = ext == "" ? ".csv" : ext
    out_path = string(root, "_15min", ext_final)

    df_new = DataFrame()
    df_new[!, "time"] = [string(t) * ".0" for t in times_15]
    for (j, feat) in enumerate(ORDERED_BASE)
        col = vec(new_mat[j, :])
        if feat in READABLE_FLOAT_FEATS
            df_new[!, feat] = col
        else
            df_new[!, feat] = Int.(col)
        end
    end

    println("→ Schreibe 15-Minuten-CSV nach $out_path …")
    CSV.write(out_path, df_new)

    println("→ Plotte Original vs. 15-Minuten …")
    traces_orig_real, traces_orig_pred = traces_for_denorm_mats(orig_mat, orig_mat; cols=ORDERED_BASE)
    traces_new_real, traces_new_pred   = traces_for_denorm_mats(new_mat, new_mat; cols=ORDERED_BASE)
    p_orig = plot(vcat(traces_orig_real, traces_orig_pred))
    p_new  = plot(vcat(traces_new_real, traces_new_pred))
    display([p_orig; p_new])

    println("→ Fertig: $out_path")
    return out_path
end



# FM-Variante: autoregressiver 500er-Run via ODE-Integration (CFM)
# nutzt integrate_cfm_euler / integrate_cfm_midpoint aus HK_model.jl
function generate_run!(model::FMTransformer, E::AbstractMatrix{<:Real};
                       run_len::Int=500, ctx::Int=20,
                       τ::Float32=1f0,                # nur zur Signatur-Kompatibilität, wird ignoriert
                       steps::Int=38, solver::Symbol=:midpoint,
                       seed::Union{Nothing,Int}=nothing)

    T = size(E, 2)
    @assert T >= ctx + run_len "Sequenz zu kurz: T=$T < ctx+run_len=$(ctx+run_len)"
    s = rand(1:(T - (ctx + run_len) + 1))

    # Seed-Kontext X: (D_IN, ctx, 1)
    X = reshape(Float32.(E[:, s:s+ctx-1]), D_IN, ctx, 1)

    # initialer Tageszeit-Slot aus letzter Kontextspalte
    s0 = X[5, end, 1]; c0 = X[6, end, 1]
    t_idx = nearest_slot_from_sc(s0, c0)

    # Ground truth-Ziele (z-normalisiert wie im Loader)
    real = Array{Float32}(undef, D_OUT, run_len)
    @inbounds real[:, :] = Float32.(E[7:6+D_OUT, s+ctx : s+ctx+run_len-1])

    pred = Array{Float32}(undef, D_OUT, run_len)

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    @inbounds for i in 1:run_len
        # Flow-Integration auf dem nächsten Tageszeit-Slot
        t_idx_next = (t_idx == SLOTS) ? 1 : (t_idx + 1)
        x̂ = if solver === :euler
            integrate_cfm_euler(model, X, t_idx_next; steps=steps, rng=rng)
        else
            integrate_cfm_midpoint(model, X, t_idx_next; steps=steps, rng=rng)
        end
        pred[:, i] = x̂

        # Nächstes Eingabe-Token bauen (Season bleibt, Tageszeit++ , numerische Kanäle = x̂)
        xin = X[:, end, 1]
        xin_next, t_idx = build_input_from_prediction(xin, x̂, t_idx)
        xin_next3 = reshape(xin_next, D_IN, 1, 1)

        # Sliding Window
        if size(X, 2) < ctx
            X = hcat(X, xin_next3)
        else
            X = cat(@view(X[:, 2:end, :]), xin_next3; dims=2)
        end
    end

    season = season_from_onehot(@view E[1:4, 1])
    return RunResult(season, 0, s, real, pred)
end



# Filtert seqs nach Season (prüft OneHot der ersten Spalte jedes Blocks)
function filter_by_season(seqs::Vector{<:AbstractMatrix}, season::AbstractString)
    out = Int[]
    @inbounds for (i, E) in enumerate(seqs)
        s = season_from_onehot(@view E[1:4, 1])
        if s == season
            push!(out, i)
        end
    end
    return out
end

"""
    collect_runs_for_season(model, seqs; season, n_runs=3, run_len=500, ctx=20, τ=1f0)

Wählt `n_runs` zufällige 500er-Ausschnitte aus `seqs` für die angegebene `season`,
generiert jeweils eine 500er-Prediction und liefert `Vector{RunResult}`.
"""
function collect_runs_for_season(model, seqs::Vector{<:AbstractMatrix};
                                 season::AbstractString,
                                 n_runs::Int=3, run_len::Int=500, ctx::Int=20, τ::Float32=1f0)
    idxs = filter_by_season(seqs, season)
    @assert !isempty(idxs) "Keine Blöcke für Season=$season gefunden."

    results = Vector{RunResult}()
    for _ in 1:n_runs
        # solange ziehen, bis ein Block lang genug ist
        rr = nothing
        for tries in 1:200
            si = rand(idxs)
            E  = seqs[si]
            if size(E,2) >= ctx + run_len
                rr = generate_run!(model, E; run_len=run_len, ctx=ctx, τ=τ)
                # seq_id sauber setzen
                # rr = RunResult(season, si, rr.start, rr.real, rr.pred)
                rr.seq_id = si
                break
            end
        end
        rr === nothing && error("Finde keinen ausreichend langen Block (ctx+run_len) in Season=$season.")
        push!(results, rr)
    end
    return results
end

"""
    traces_for_run(rr; cols=ORDERED_BASE)

Erzeugt pro Feature zwei Scatter-Traces (real/pred) über 1:run_len.
Gibt `Vector{AbstractTrace}` zurück (Länge = 2*D_OUT).
"""
function traces_for_run(rr::RunResult; cols=ORDERED_BASE)
    L = size(rr.real, 2)
    x = 1:L
    traces_real = AbstractTrace[]
    traces_pred = AbstractTrace[]
    @inbounds for j in 1:length(cols)
        name = cols[j]
        push!(traces_real, scatter(; x=x, y=vec(rr.real[j, :]), mode="lines", name=name))
        push!(traces_pred, scatter(; x=x, y=vec(rr.pred[j, :]), mode="lines", name=name))
    end
    return traces_real, traces_pred
end

# ---- Denormalisierung ---------------------------------------------------------
# Erwartet: stats_year[serial][feat] => (mu, sigma)
# Fallback via get_stats_year(...): (mu=0, sigma=1), falls Eintrag fehlt.

# liefert (real_denorm, pred_denorm) als Float32-Matrizen
function denorm_mats(rr::RunResult, serial::AbstractString, stats_year;
                     cols=ORDERED_BASE,
                     norm::Symbol=:year_log1p,
                     base_dir::Union{Nothing,AbstractString}=nothing,
                     seasons=("Winter","Fruehling","Sommer","Herbst"),
                     stats_path_year::Union{Nothing,AbstractString}=nothing)
    D = length(cols)
    L = size(rr.real, 2)
    real_d = Array{Float32}(undef, D, L)
    pred_d = Array{Float32}(undef, D, L)
    if norm === :year_log1p
        base_dir === nothing && error("base_dir required for :year_log1p denormalization")
        stats_path = stats_path_year === nothing ? joinpath(base_dir, "stats_by_serial_year.json") : stats_path_year
        lognorm = build_log1p_norm(base_dir, String(serial);
                                   seasons=seasons,
                                   stats_year=stats_year,
                                   stats_path_year=stats_path)
        @inbounds for j in 1:D
            s = lognorm.scale[j]
            μz = lognorm.mu_z[j]
            σz = max(lognorm.sig_z[j], 1f-6)
            @views real_d[j, :] .= s .* (expm1.(rr.real[j, :] .* σz .+ μz))
            @views pred_d[j, :] .= s .* (expm1.(rr.pred[j, :] .* σz .+ μz))
            real_d[j, :] .= max.(real_d[j, :], 0f0)
            pred_d[j, :] .= max.(pred_d[j, :], 0f0)
        end
    else
        @inbounds for j in 1:D
            feat = cols[j]
            st   = get_stats_year(stats_year, String(serial), String(feat))
            μ    = Float32(st.mu)
            σ    = Float32(max(st.sigma, 1e-6))
            @views real_d[j, :] .= rr.real[j, :] .* σ .+ μ
            @views pred_d[j, :] .= rr.pred[j, :] .* σ .+ μ
        end
    end
    return real_d, pred_d
end

# Baut Traces aus *denormalisierten* Matrizen
function traces_for_denorm_mats(real_d::AbstractMatrix, pred_d::AbstractMatrix; cols=ORDERED_BASE, colors::Union{Nothing,Vector{String}}=nothing)
    L = size(real_d, 2)
    x = 1:L
    D = length(cols)

    # Farben vorbereiten (zyklisch über Palette)
    if colors === nothing
        colors = [PLOTLY13[mod1(j, length(PLOTLY13))] for j in 1:D]
    else
        @assert length(colors) >= D "colors muss mind. so lang wie cols sein"
    end

    traces_real = AbstractTrace[]
    traces_pred = AbstractTrace[]
    @inbounds for j in 1:D
        cname = cols[j]
        col   = colors[j]

        if cname in ("power_mean_kw","power_max_kw","power_min_kw",
                    "power_avail_wind_mean_kw",
                    "power_avail_tech_mean_kw",
                    "power_avail_force_maj_mean_kw",
                    "power_avail_ext_mean_kw",)
            yreal = vec(real_d[j, :]) ./100
            ypred = vec(pred_d[j, :]) ./ 100
        else
            yreal = vec(real_d[j, :])
            ypred = vec(pred_d[j, :])
        end

        # gleiche Farbe, optional legendgroup für saubere Legenden-Gruppierung
        push!(traces_real, scatter(; x=x, y=yreal,
                                   mode="lines", name="$cname real",
                                   line=attr(color=col), legendgroup=cname))
        push!(traces_pred, scatter(; x=x, y=ypred,
                                   mode="lines", name="$cname pred",
                                   line=attr(color=col), legendgroup=cname))
    end
    return traces_real, traces_pred
end

"""
    eval_collect_all(model; serial="1011089", base="HK_blocks", ctx=20,
                     run_len=500, n_runs=3, τ=1f0, norm=:year_log1p,
                     seasons=("Winter","Fruehling","Sommer","Herbst"))

Lädt `seqs` für `serial`, sammelt pro Season drei 500er-Runs und baut je Run Trace-Arrays.
Return:
- `results_by_season :: Dict{String, Vector{RunResult}}`
- `traces_by_season  :: Dict{String, Vector{Vector{AbstractTrace}}}`
"""
function eval_collect_all(model=model;
                          serial::AbstractString=SERIAL,
                          base::AbstractString=BASE,
                          ctx::Int=CTX, run_len::Int=500, n_runs::Int=3,
                          τ::Float32=1f0, norm::Symbol=:year_log1p,
                          stats_path_year::AbstractString=joinpath(base, "stats_by_serial_year.json"),
                          seasons=("Winter","Fruehling","Sommer","Herbst"))
    # Daten + Jahres-Stats laden
    seqs = load_blocks_for_serial(base, serial;
                                  norm=norm,
                                  seasons=seasons,
                                  stats_path_year=stats_path_year)
    stats_year = load_stats_year(stats_path_year)

    global results_by_season = Dict{String, Vector{RunResult}}()

    global traces_real_by_season  = Dict{String, Vector{Vector{AbstractTrace}}}()
    global traces_pred_by_season  = Dict{String, Vector{Vector{AbstractTrace}}}()

    for s in SEASON_NAMES
        runs = collect_runs_for_season(model, seqs; season=s, n_runs=n_runs, run_len=run_len, ctx=ctx, τ=τ)
        results_by_season[s] = runs


        trsets_real = Vector{Vector{AbstractTrace}}()
        trsets_pred = Vector{Vector{AbstractTrace}}()

        for r in runs
            # <- Denormalisierung pro Run
            real_d, pred_d = denorm_mats(r, serial, stats_year;
                                         cols=ORDERED_BASE,
                                         norm=norm,
                                         base_dir=base,
                                         seasons=seasons,
                                         stats_path_year=stats_path_year)
            traces_real, traces_pred = traces_for_denorm_mats(real_d, pred_d; cols=ORDERED_BASE)
            push!(trsets_real, traces_real)
            push!(trsets_pred, traces_pred)
        end

        traces_real_by_season[s] = trsets_real
        traces_pred_by_season[s] = trsets_pred

        p1 = plot(trsets_real[1])
        p2 = plot(trsets_pred[1])
        p = [p1;p2]

        layout = Layout(
        title="$s, Block $(runs[1].seq_id), line $(runs[1].start)",
        )

        relayout!(p, layout.fields)
        display(p)
    end

    
end
