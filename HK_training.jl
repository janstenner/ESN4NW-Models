using Zygote
using FileIO, JLD2

# HK_training.jl
#
# Setzt das Training für den autoregressiven Transformer auf.
# - nutzt HK_model.jl (Decoder-only Transformer mit gaußschem Kopf)
# - nutzt HK_loader.jl (CSV -> Embeddings -> Fenster -> Minibatches)
#
const CTX    = 20
const SERIAL = "1011089"
const BASE   = "HK_blocks"
const BATCH  = 64
const LR     = 1f-3
const EPOCHS = 1
const SHUFFLE = true
const NHEADS_WARMUP = 0          # optional: nicht genutzt, Platzhalter
const SEED = Int(floor(rand()*100000))

const SAVE_DIR_AR = "./saves_HK"
const SAVE_DIR_FM = "./saves_HK_FM"


# --------------------------
# Hyperparameter Model
# --------------------------
const D_IN   = 19          # 4 (Season one-hot) + 2 (sin/cos Zeit) + 13 numerische Kanäle
const D_OUT  = 13          # Ziel-Dimension: deine 13 kontinuierlichen Kanäle für t+1
const NUM_IDX = 7:(6 + D_OUT)   # die 13 numerischen Kanäle im D_IN-Token
const D_MODEL= 64
const N_HEAD = 4
const N_LAY  = 4
const D_FF   = 128
const DROPOUT= 0.1
const RANK   = 4           # Low-rank-Kovarianz-Rang
const MAX_CTX = 30

using Flux, LinearAlgebra, Random, Statistics

# ---------------------------------------------------------------------------
# Includes (liegen im selben Ordner wie diese Datei)
# ---------------------------------------------------------------------------
include("HK_model.jl")
include("HK_loader.jl")
include("HK_eval.jl")



# --- Helfer: nächster Zeit-Slot aus sin/cos bestimmen (TIME_LUT/SLOTS aus HK_loader) ---
@inline function nearest_slot_from_sc(s::Real, c::Real)::Int
    bestk = 1
    bestd = typemax(Float32)
    @inbounds for k in 1:SLOTS
        ds = Float32(TIME_LUT[1,k]) - Float32(s)
        dc = Float32(TIME_LUT[2,k]) - Float32(c)
        d  = ds*ds + dc*dc
        if d < bestd; bestd = d; bestk = k; end
    end
    return bestk
end



# --- CFM-Schedules (Gaussian CFM) ---------------------------------------------
const SIGMA_MIN = 1f-3     # kannst du z.B. auf 1e-2/1e-4 variieren
@inline alpha(t::Float32)    = t
@inline alphadot(t::Float32) = 1f0
@inline sigma(t::Float32)    = exp(log(SIGMA_MIN)*t)        # = SIGMA_MIN^t
@inline sigmadivdot(t::Float32) = log(SIGMA_MIN)            # d/dt log sigma(t)


# --- CFM: Batch vorbereiten (X_aug, t_batch, u_cond) ---------------------------
function prepare_fm_augmented_batch(X::AbstractArray{<:Real,3}, Y::AbstractArray{<:Real,3};
                                    ctx::Int=CTX)
    D_in, ctxT, B = size(X); @assert ctxT == ctx
    X_aug   = Array{Float32}(undef, D_in, ctx+1, B)
    X_aug[:, 1:ctx, :] .= Float32.(X)
    t_batch = rand(Float32, B)                    # t ~ U[0,1] pro Sample
    u_cond  = Array{Float32}(undef, D_OUT, B)    # Ziel-Geschwindigkeit

    @inbounds for b in 1:B
        x1 = vec(@view Y[:, size(Y, 2), b]) # nächster Datenvektor (z-normalisiert)
        x1 = Float32.(x1)

        # Gaussian CFM: x_t = mu_t + sigma(t) * eps
        t  = t_batch[b]
        μt = alpha(t) .* x1
        σt = sigma(t)
        ε  = randn(Float32, D_OUT)
        x_t = μt .+ σt .* ε

        # Zielgeschwindigkeit: u_t = (σ̇/σ) (x_t - μ_t) + μ̇_t
        u_cond[:, b] = sigmadivdot(t) .* (x_t .- μt) .+ alphadot(t) .* x1

        # Zeit-Slot für den *nächsten* Schritt
        s0   = X[5, end, b]; c0 = X[6, end, b]
        tidx = nearest_slot_from_sc(s0, c0)
        tidx_next = (tidx == SLOTS) ? 1 : (tidx + 1)

        # Flow-Token anhängen
        flow_vec = Vector{Float32}(undef, D_IN)
        XX = @view X[:, end, b]
        make_flow_token_from_template!(flow_vec, XX, x_t, tidx_next)
        X_aug[:, ctx+1, b] = flow_vec
    end

    return X_aug, t_batch, u_cond
end





# Sanity: D_IN/D_OUT kommen aus HK_model.jl; ORDERED_BASE etc. aus HK_loader.jl

# ---------------------------------------------------------------------------
# Low-rank+diag NLL (Woodbury) über gesamte Sequenz
# Σ = U Uᵀ + diag(σ²), mit U:(D, r), σ = exp(logσ)
# Für jede (t,b):   δᵀ Σ⁻¹ δ = δᵀ D⁻¹ δ - (UᵀD⁻¹δ)ᵀ (I + UᵀD⁻¹U)⁻¹ (UᵀD⁻¹δ)
# logdet Σ = ∑ 2 log σ_i + logdet(I + UᵀD⁻¹U)
# ---------------------------------------------------------------------------
@inline function nll_mvg_lowrank_seq_woodbury(mu, logσ, U, Y; ϵ=1f-6)
    # mu, logσ : (D_OUT, T, B)
    # U        : (D_OUT, RANK, T, B)
    # Y        : (D_OUT, T, B)
    D, T, B = size(mu)
    total = 0.0f0
    @inbounds for b in 1:B, t in 1:T
        μt   = view(mu, :, t, b)
        lσt  = view(logσ, :, t, b)
        σt   = exp.(lσt) .+ 1f-6
        invD = 1f0 ./ (σt .* σt)                 # D^{-1} diagonal als Vektor

        δ    = view(Y, :, t, b) .- μt            # (D,)
        Ut   = view(U, :, :, t, b)               # (D, r)

        # Woodbury-Terme
        w    = invD .* δ                         # D^{-1} δ
        M    = transpose(Ut) * w                 # (r,)
        G    = transpose(Ut) * (invD .* Ut)      # (r,r)
        A    = G + I                      # (r,r), identitätsverschoben

        F    = cholesky(Symmetric(A) + ϵ*I)
        α    = F \ M

        mahal = dot(δ, w) - dot(M, α)
        ldetD = 2f0 * sum(log.(σt))
        ldetA = 2f0 * sum(log.(diag(F.L)))
        total += 0.5f0 * (mahal + (ldetD + ldetA) + D*log(2f0*π))
    end
    return total / (T*B)
end

# Optional: einfache Cholesky-Variante (direkt auf Σ), gut für D=13
function nll_mvg_lowrank_seq_chol(mu, logσ, U, Y; ϵ=1f-6)
    D, T, B = size(mu)
    total = 0.0f0
    @inbounds for b in 1:B, t in 1:T
        μt   = view(mu, :, t, b)
        lσt  = view(logσ, :, t, b)
        σt   = exp.(lσt) .+ 1f-6
        Ut   = view(U, :, :, t, b)
        Σ    = Ut*Ut' .+ Diagonal(σt.^2)
        F    = cholesky(Symmetric(Σ + ϵ*I))
        δ    = view(Y, :, t, b) .- μt
        α    = F \ δ
        total += 0.5f0 * ( (2f0*sum(log.(diag(F.L))))*1f0 + dot(δ, α) + D*log(2f0*π) )
    end
    return total / (T*B)
end

# Ein einheitlicher Wrapper (wähle Woodbury als Default)
const USE_WOODBURY = true
nll_sequence(mu, logσ, U, Y) = USE_WOODBURY ?
    nll_mvg_lowrank_seq_woodbury(mu, logσ, U, Y) :
    nll_mvg_lowrank_seq_chol(mu, logσ, U, Y)


    

# --- Training: verzweigt nach Modelltyp ---------------------------------------
function train!(model; epochs::Int=EPOCHS, lr::Float32=LR, batch::Int=BATCH, ctx::Int=CTX,
                serial::AbstractString=SERIAL, base::AbstractString=BASE,
                shuffle::Bool=SHUFFLE, seed::Int=SEED)

    Random.seed!(seed)
    seqs   = load_blocks_for_serial(base, serial; norm=:year)
    loader = make_loader(seqs; ctx=ctx, batchsize=batch, shuffle=shuffle)

    opt    = Flux.setup(Flux.AdamW(lr), model)

    total_batches = length(loader)

    function _log(ep, it, loss)
        @info "epoch=$(ep) iter=$(it)/$(total_batches) loss=$(round(loss, digits=5))"
    end

    global losses
    for ep in 1:epochs
        it = 0
        if shuffle
            println("Re-Shuffle!")
            Random.shuffle!(loader.idx)
        end

        for (X, Y) in loader
            it += 1
            if model isa ARTransformer
                # --- autoregressives Training: NLL über gesamte Sequenz ---
                gs = Flux.gradient(model) do m
                    μ, logσ, U = m(X)                    # (D_OUT,T,B), ...
                    nll = nll_sequence(μ, logσ, U, Y)    # mittelt über T*B
                    Zygote.ignore_derivatives() do
                        push!(losses, Float32(nll))
                    end
                    nll
                end
                Flux.update!(opt, model, gs[1])

            elseif model isa FMTransformer
                # --- CFM-Training: MSE zwischen vθ(X_aug,t) und u_cond ---
                X_aug, t_batch, u_cond = prepare_fm_augmented_batch(X, Y; ctx=ctx)
                gs = Flux.gradient(model) do m
                    v_pred = m(X_aug, t_batch)           # (D_OUT, B)
                    l = mean((v_pred .- u_cond).^2)      # MSE über D_OUT,B
                    Zygote.ignore_derivatives() do
                        push!(losses, Float32(l))
                    end
                    l
                end
                Flux.update!(opt, model, gs[1])

            else
                error("Unbekannter Modelltyp: $(typeof(model))")
            end

            (it % 50 == 0) && _log(ep, it, mean(losses[end-49:end]))
        end
        @info "epoch=$(ep) done."
    end
    return model
end




# model = ARTransformer()
model = FMTransformer()
losses = Float32[]


# @info "Start training" D_IN D_OUT CTX SERIAL EPOCHS BATCH LR
# train!(model)




function save_model(; serial=SERIAL)
    @assert @isdefined(model)  "save_model: `model` ist nicht definiert."
    losses_ = @isdefined(losses) ? losses : Float32[]

    dir = if model isa FMTransformer
        SAVE_DIR_FM
    elseif model isa ARTransformer
        SAVE_DIR_AR
    else
        error("save_model: Unbekannter Modelltyp: $(typeof(model))")
    end

    isdir(dir) || mkpath(dir)
    path = joinpath(dir, string(serial, ".jld2"))

    FileIO.save(path,
        "model",  model,
        "losses", losses_,
    )
    return path
end


function load_model(path = SAVE_DIR_FM; serial=SERIAL)

    path = path * string("/", serial, ".jld2")

    data = FileIO.load(path)
    global model  = data["model"]
    global losses = get(data, "losses", Float32[])  # falls älterer Save ohne losses

    return (path=path, mtype=typeof(model))
end


function plot_losses(smoothing = 50)
    to_plot = Float32[]
    for i in smoothing:length(losses)
        push!(to_plot, mean(losses[i+1-smoothing:i]))
    end

    p = plot(to_plot)
    display(p)
end


# Gesamtzahl der trainierbaren Parameter
function count_params(m = model)
    s = 0
    for p in Flux.params(m)      # iteriert über eindeutige Arrays
        s += length(p)
    end
    return s
end

# Kleines Breakdown (optional)
function param_breakdown(m = model)
    sizes = map(size, Flux.params(m))
    lens  = map(length, Flux.params(m))
    return (; total=sum(lens), arrays=length(lens), sizes=sizes, lengths=lens)
end