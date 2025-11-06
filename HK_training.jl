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
        make_flow_token_from_template!(flow_vec, @view X[:, end, b], x_t, tidx_next)
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

            (it % 50 == 0 || it == 1) && _log(ep, it, losses[end])
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




function load_model(serial = SERIAL)

    data = FileIO.load("./saves_HK/$(serial).jld2")
    global model = data["model"]
    global losses = data["losses"]

end

function save_model()
    isdir("./saves_HK") || mkdir("./saves_HK")


    FileIO.save("./saves_HK/$(SERIAL).jld2",
        "model", model,
        "losses", losses
    )

end