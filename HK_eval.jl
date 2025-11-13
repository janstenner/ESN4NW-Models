

using Random, LinearAlgebra
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
                     norm::Symbol=:year,
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
                     run_len=500, n_runs=3, τ=1f0, norm=:year,
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
                          τ::Float32=1f0, norm::Symbol=:year,
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
