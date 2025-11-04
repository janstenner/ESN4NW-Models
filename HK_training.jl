using Zygote
using FileIO, JLD2

# HK_training.jl
#
# Setzt das Training für den autoregressiven Transformer auf.
# - nutzt HK_model.jl (Decoder-only Transformer mit gaußschem Kopf)
# - nutzt HK_loader.jl (CSV -> Embeddings -> Fenster -> Minibatches)
#
# You edit here:
const CTX    = 20
const SERIAL = "1011089"
const BASE   = "HK_blocks"
const BATCH  = 64
const LR     = 1f-3
const EPOCHS = 1
const SHUFFLE = true
const NHEADS_WARMUP = 0          # optional: nicht genutzt, Platzhalter
const SEED = Int(floor(rand()*100000))

using Flux, LinearAlgebra, Random, Statistics

# ---------------------------------------------------------------------------
# Includes (liegen im selben Ordner wie diese Datei)
# ---------------------------------------------------------------------------
include("HK_model.jl")
include("HK_loader.jl")

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

# ---------------------------------------------------------------------------
# Training: Loader -> Model -> NLL über Sequenz -> Update
# ---------------------------------------------------------------------------
function train!(model; epochs::Int=EPOCHS, lr::Float32=LR, batch=BATCH, ctx::Int=CTX,
                serial::AbstractString=SERIAL, base::AbstractString=BASE,
                shuffle::Bool=SHUFFLE, seed::Int=SEED)

    Random.seed!(seed)

    # Daten laden (Ganzjahresnormierung pro Serial; Seasons werden automatisch durchsucht)
    seqs = load_blocks_for_serial(base, serial; norm=:year)

    # Minibatches aus Fenstern
    loader = make_loader(seqs; ctx=ctx, batchsize=batch, shuffle=shuffle)

    # Optimizer
    opt = Flux.setup(Flux.AdamW(lr), model)

    # kleines Logging
    total_batches = length(loader)
    function _log(ep, it, loss)
        @info "epoch=$(ep) iter=$(it)/$(total_batches) loss=$(round(loss, digits=5))"
    end

    for ep in 1:epochs
        it = 0
        for (X, Y) in loader
            it += 1
            losses = Float32[]
            # Vorwärts + Loss (über alle Zeitpositionen)
            gs = Flux.gradient(model) do m
                μ, logσ, U = m(X)            # (D_OUT,T,B), (D_OUT,T,B), (D_OUT,RANK,T,B)
                nll = nll_sequence(μ, logσ, U, Y)  # mittelt über T*B

                Zygote.ignore_derivatives() do
                    push!(losses, mean(nll))
                end

                nll
            end
            Flux.update!(opt, model, gs[1])

            (it % 50 == 0 || it == 1) && _log(ep, it, losses[end])
        end
        # Epoch-Summary (einfacher Running-Loss − hier: letzter Minibatch)
        @info "epoch=$(ep) done."
    end

    return model
end




model = ARTransformer()
@info "Start training" D_IN D_OUT CTX SERIAL EPOCHS BATCH LR
# train!(model)




function load_model(serial = SERIAL)

    global model = FileIO.load("/saves_HK/$(serial).jld2","model")

end

function save_model()
    isdir("/saves_HK") || mkdir("/saves_HK")


    FileIO.save("/saves_HK/$(SERIAL).jld2","model",model)

end