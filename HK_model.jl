using Flux, NNlib, LinearAlgebra, Statistics



# --------------------------
# Hilfen: Masken & Blöcke
# --------------------------

# === Zeit-LUT vorbereiten (2 x 144) ==========================================
const SLOTS  = 144
const Δθ     = 2f0 * π / Float32(SLOTS)

# TIME_LUT[1, k] = sin(θ_k), TIME_LUT[2, k] = cos(θ_k), k=1..144
const TIME_LUT = let M = Matrix{Float32}(undef, 2, SLOTS)
    @inbounds for k in 1:SLOTS
        s, c = sincos(Float32(k-1) * Δθ)
        M[1,k] = s
        M[2,k] = c
    end
    M
end

# robustes 1-basiertes Modul (1..SLOTS)
@inline next_slot(k) = k == SLOTS ? 1 : k + 1

# x kann (D_IN, T) oder (T, D_IN) oder Vector{<:AbstractVecOrMat} sein:
function ensure_3d(x; d_in::Int)
    if x isa AbstractVector  # z.B. Vector{Matrix}
        X = reduce(hcat, x)                  # (d_in, T) oder (T, d_in)
    else
        X = x
    end
    size(X,1) == d_in || (X = permutedims(X))  # bringe auf (d_in, T)
    ndims(X) == 2 && (X = reshape(X, size(X,1), size(X,2), 1))  # → (d_in, T, 1)
    return X
end


# Causale Maske: KVxQxHeadsxBatch
make_causal(kv_len::Int, q_len::Int, nheads::Int, batch::Int) =
    reshape(NNlib.make_causal_mask(zeros(Bool, kv_len, q_len)), kv_len, q_len, 1, 1) .|> x->x ? 1f0 : 0f0 .+ zeros(Float32, 0) # ensure Float32
# (Flux erwartet mask als Array broadcastbar auf (kv_len, q_len, nheads, batch))

# Transformer-Decoder-Block (Self-Attn + FFN)
struct DecoderBlock
    mha::MultiHeadAttention
    ln1::LayerNorm
    ln2::LayerNorm
    ff::Chain
    dropout::Dropout
end

function DecoderBlock(d_model::Int, nheads::Int, d_ff::Int; pdrop=0.1)
    DecoderBlock(
        MultiHeadAttention(d_model, nheads=nheads, dropout_prob=pdrop),
        LayerNorm(d_model),
        LayerNorm(d_model),
        Chain(Dense(d_model, d_ff, gelu), Dropout(pdrop), Dense(d_ff, d_model)),
        Dropout(pdrop)
    )
end

# Vorwärts für einen Block (mit causal mask)
function (m::DecoderBlock)(x, mask)
    # x: (d_model, T, B)
    y, _ = m.mha(x, x, x; mask=mask)   # Self-Attention
    x = m.ln1(x .+ m.dropout(y))
    z = m.ff(x)
    x = m.ln2(x .+ m.dropout(z))
    return x
end

# --------------------------
# Modell: Eingangsprojektion -> N Decoder-Blöcke -> Gauß-Kopf
# --------------------------
struct ARTransformer
    inproj::Dense
    blocks::Vector{DecoderBlock}
    ln_final::LayerNorm
    # Gauß-Kopf Parameterizer:
    #  - μ:       Dense(d_model, D_OUT)
    #  - logσ:    Dense(d_model, D_OUT)
    #  - U:       Dense(d_model, D_OUT*RANK)  (reshape zu D_OUT×RANK)
    mu_head::Dense
    logstd_head::Dense
    u_head::Dense
end

function ARTransformer()
    blocks = [DecoderBlock(D_MODEL, N_HEAD, D_FF; pdrop=DROPOUT) for _ in 1:N_LAY]
    ARTransformer(
        Dense(D_IN, D_MODEL), blocks, LayerNorm(D_MODEL),
        Dense(D_MODEL, D_OUT),
        Dense(D_MODEL, D_OUT),
        Dense(D_MODEL, D_OUT*RANK)
    )
end

# Vorwärts: gesamte Sequenz -> Parameter für t+1 auf dem letzten Zeitindex
function (m::ARTransformer)(x)

    x = ensure_3d(x; d_in=D_IN)          # ← neu, macht aus 2D/Vector → 3D


    # x: (D_IN, T, B)  〈— deine eingebetteten Features pro Schritt
    h = m.inproj(x)            # (D_MODEL, T, B)
    T = size(h, 2); B = size(h, 3)
    mask = repeat(NNlib.make_causal_mask(zeros(Bool, T, T)), 1, 1, N_HEAD, B) # (T,T,Heads,B), Bool

    # MultiHeadAttention will (kv_len,q_len,heads,batch); unsere (T,T,…) passt
    for blk in m.blocks
        h = blk(h, mask)
    end
    h = m.ln_final(h)          # (D_MODEL, T, B)

    H      = reshape(h, D_MODEL, T*B)                                    # (D_MODEL, T*B)
    MU_2d  = m.mu_head(H)                                                 # (D_OUT,   T*B)
    LOG_2d = m.logstd_head(H)                                             # (D_OUT,   T*B)
    U_2d   = m.u_head(H)                                                  # (D_OUT*RANK, T*B)

    mu     = reshape(MU_2d,  D_OUT, T, B)                                 # (D_OUT, T, B)
    logσ   = reshape(LOG_2d, D_OUT, T, B)                                 # (D_OUT, T, B)
    U      = reshape(U_2d,   D_OUT, RANK, T, B)                           # (D_OUT, RANK, T, B)

    return mu, logσ, U
end

# --------------------------
# NLL: multivariate Gauß, Kovarianz Σ = U Uᵀ + diag(σ²)
# --------------------------
function nll_mvg_lowrank(mu, logσ, U, y)
    # mu:   (D_OUT,B), logσ: (D_OUT,B), U: (D_OUT,RANK,B), y: (D_OUT,B)
    B = size(mu, 2)
    loss = 0.0
    @inbounds for b in 1:B
        μb   = view(mu, :, b)
        logσb= view(logσ, :, b)
        σb   = exp.(logσb) .+ 1f-6             # jitter
        Ub   = view(U, :, :, b)                # (D_OUT,RANK)
        Σb   = Ub*Ub' .+ Diagonal(σb.^2)       # (D_OUT,D_OUT)
        # Cholesky für logdet & Lösung
        F    = cholesky(Symmetric(Σb + 1f-6I))
        δ    = (y[:, b] .- μb)
        α    = F \ δ
        loss += 0.5f0 * (logdet(F) * 2f0 + dot(δ, α) + D_OUT*log(2f0*π))
    end
    return loss / B
end

# --------------------------
# Mini-Train-Step (Teacher Forcing 1-Schritt)
# --------------------------
# batch_x: (D_IN, T, B), batch_y: (D_OUT, B)  —> y ist Ziel für t+1
function loss_fun(model, batch_x, batch_y)
    μ, logσ, U = model(batch_x)
    nll_mvg_lowrank(μ, logσ, U, batch_y)
end

# Beispiel-Optimierer
# model = ARTransformer()
# opt = Flux.setup(Flux.AdamW(1e-3), model)

# Dummy-Schleife (du ersetzt batch_x/batch_y mit deinem DataLoader)
# for (batch_x, batch_y) in loader
#     gs = Flux.gradient(model) do m
#         loss_fun(m, batch_x, batch_y)
#     end
#     Flux.update!(opt, model, gs)
# end

# --------------------------
# Sampler (autoregressiv)
# --------------------------
function sample_autoregressive(model::ARTransformer = model, x0 = nothing; steps::Int = 30)
    # x0: initiale Sequenz (D_IN, T0, 1) mit deinen Features bis t0 (inkl. Season+Zeit+Kanäle)
    if isnothing(x0)
        x0 = zeros(Float32,D_IN,1,1)
        x0[1] = 1.0 # season Winter
        t_idx = 1
        x0[5:6] = TIME_LUT[:,t_idx]
    end

    x = x0
    ysamp = Matrix{Float32}(undef, D_IN, steps)
    for s in 1:steps
        μ, logσ, U = model(x)                 # Parameter für t+1
        σ  = exp.(logσ) .+ 1f-6
        Σ  = U[:,:,1]*U[:,:,1]' .+ Diagonal(σ[:,1].^2)
        F  = cholesky(Symmetric(Σ + 1f-6I))
        # Ziehe y_{t+1}
        ϵ  = randn(Float32, D_OUT)
        yt = μ[:,1] .+ F.L * ϵ
        

        
        xin_next, t_idx = build_input_from_prediction(x[:, end, 1], yt, t_idx) 
        ysamp[:, s] = xin_next
        x = hcat(x, reshape(xin_next, D_IN, 1, 1))

        T = size(x, 2)
        if T > MAX_CTX
            x = @view x[:, T - MAX_CTX + 1:T, :]       # SubArray-View: keine Kopie
        end
    end
    return ysamp
end


function build_input_from_prediction(x_last::AbstractVector{<:Real},
                                     y_pred::AbstractVector{<:Real},
                                     t_idx::Int; step_slots::Int=1)
    @assert length(x_last) == D_IN
    @assert length(y_pred) == D_OUT
    # nächsten Slot bestimmen (10 Minuten = 1 Slot)
    k = t_idx
    @inbounds for _ in 1:step_slots
        k = next_slot(k)
    end

    out = Vector{Float32}(undef, D_IN)

    # 1) Season kopieren (1:4)
    @inbounds for i in 1:4
        out[i] = Float32(x_last[i])
    end
    # 2) Zeit aus LUT (5:6)
    @inbounds begin
        out[5] = TIME_LUT[1, k]  # sin
        out[6] = TIME_LUT[2, k]  # cos
    end
    # 3) 13 Kanäle (7:19)
    @inbounds for i in 1:D_OUT
        out[6+i] = Float32(y_pred[i])
    end

    return out, k  # gib neuen Slot-Index zurück
end








#-------- Flow Matching -------

# --- t-Embedding --------------------------------------------------------------
struct TEmbedding
    B::Vector{Float32}      # Frequenzen b_k (L)
    proj::Chain             # MLP: (2L) -> D_MODEL
end

# nur proj ist trainierbar
Flux.@layer TEmbedding
Flux.trainable(te::TEmbedding) = (proj = te.proj,)

function TEmbedding(d_model = D_MODEL; bands::Int=16, sigma::Float32=16f0)
    B = randn(Float32, bands) .* sigma
    proj = Chain(
        Dense(2*bands, D_MODEL, x -> x .* (1f0 ./ (1f0 .+ exp.(-x)))),  # SiLU
        Dense(D_MODEL, D_MODEL)
    )
    return TEmbedding(B, proj)
end

# t: Vector{Float32} der Länge BATCH oder scalar -> (D_MODEL, 1, BATCH)
function (te::TEmbedding)(t::AbstractVector{<:Real})
    t32  = Float32.(t)
    L    = length(te.B)
    Bsz  = length(t32)
    Bcol = reshape(te.B, L, 1)        # (L,1)
    trow = reshape(t32, 1, Bsz)       # (1,B)
    ω    = (2f0 * π) .* (Bcol .* trow)     # (L,B)
    F    = vcat(sin.(ω), cos.(ω))          # (2L,B) — reine Broadcasts, keine Mutation
    Z    = te.proj(F)                       # (D_MODEL,B)
    return reshape(Z, size(Z,1), 1, size(Z,2))  # (D_MODEL,1,B)
end

# --- Token-Type-Bias (statt Embedding) ---------------------------------------
struct TypeBias
    ctx::Vector{Float32}   # (D_MODEL,)
    flow::Vector{Float32}  # (D_MODEL,)
end
Flux.@layer TypeBias     # macht die Felder trainierbar

TypeBias(d_model = D_MODEL) = TypeBias(
    0.02f0 .* randn(Float32, d_model),
    0.02f0 .* randn(Float32, d_model)
)


struct FMTransformer
    inproj::Dense
    te::TEmbedding
    ttype::TypeBias
    blocks::Vector{DecoderBlock}
    ln_final::LayerNorm
    head::Dense
end

function FMTransformer()
    blocks = [DecoderBlock(D_MODEL, N_HEAD, D_FF; pdrop=DROPOUT) for _ in 1:N_LAY]
    FMTransformer(
        Dense(D_IN, D_MODEL),
        TEmbedding(),
        TypeBias(),
        blocks,
        LayerNorm(D_MODEL),
        Dense(D_MODEL, D_OUT)
    )
end


function (m::FMTransformer)(x,t)

    x = ensure_3d(x; d_in=D_IN)          # ← neu, macht aus 2D/Vector → 3D


    # x: (D_IN, T, B)  〈— deine eingebetteten Features pro Schritt
    h = m.inproj(x)            # (D_MODEL, T, B)


    # 1) Token-Typ addieren
    # h: (D_MODEL, T, B)
    flow_col = size(h, 2); B = size(h, 3)

    # Type-Embeddings als (D_MODEL,1,1)
    type_ctx  = reshape(m.ttype.ctx,  D_MODEL, 1, 1)  # (D_MODEL,1,1)
    type_flow = reshape(m.ttype.flow, D_MODEL, 1, 1)  # (D_MODEL,1,1)
    type_add = cat(repeat(type_ctx,  1, flow_col-1, B),
                   repeat(type_flow, 1, 1,   B); dims=2)

    h = h .+ type_add

    # 2) t-Embedding nur aufs Flow-Token
    t_emb = m.te(t)  # (D_MODEL, 1, B)
    t_add = cat(zeros(Float32, D_MODEL, flow_col-1, B), t_emb; dims=2)  # (D_MODEL, T, B)
    h = h .+ t_add

    T = size(h, 2); B = size(h, 3)
    mask = repeat(NNlib.make_causal_mask(zeros(Bool, T, T)), 1, 1, N_HEAD, B) # (T,T,Heads,B), Bool

    # MultiHeadAttention will (kv_len,q_len,heads,batch); unsere (T,T,…) passt
    for blk in m.blocks
        h = blk(h, mask)
    end
    h = m.ln_final(h)          # (D_MODEL, T, B)


    output = m.head(Array(@view h[:, flow_col, :]))         # (D_OUT, B); copy vermeidet SubArray-Mutation
    return output
end


# Baut aus dem letzten Kontext-Token ein Flow-Token, ersetzt NUR die numerischen 13 Werte
# und setzt die Tageszeit auf den vorgegebenen Slot t_idx (Season-OneHot bleibt vom Template).
function make_flow_token_from_template!(dest::AbstractVector{Float32},
                                        template::AbstractVector{<:Real},
                                        x_numeric::AbstractVector{<:Real},
                                        t_idx::Int)
    @assert length(dest) == D_IN
    @assert length(template) == D_IN
    @assert length(x_numeric) == D_OUT
    @inbounds begin
        # kopiere Season + alles
        @views dest[:] = Float32.(template[:])
        # setze Tageszeit (Flow-Integration: physische Tageszeit bleibt fix)
        dest[5] = TIME_LUT[1, t_idx]   # sin
        dest[6] = TIME_LUT[2, t_idx]   # cos
        # ersetze numerische Kanäle durch aktuellen Flow-Zustand x_t
        @views dest[NUM_IDX] .= Float32.(x_numeric)
    end
    return dest
end

# Ruft v_theta(x_t, t | Kontext) ab.
# X_ctx: (D_IN, ctx, 1)    — dein 20er Fenster ohne Flow-Token
# x_t  : (D_OUT,)          — aktueller Flow-Zustand (z-normalisiert)
# t    : Float32 in [0,1]  — Flow-Zeit
# t_idx: Int               — fixer Tageszeit-Slot des nächsten realen Zeitschritts
function fm_velocity(model, X_ctx::AbstractArray{<:Real,3},
                     x_t::AbstractVector{<:Real}, t::Float32, t_idx::Int)::Vector{Float32}
    @assert size(X_ctx, 3) == 1
    flow_vec = Vector{Float32}(undef, D_IN)
    # Vorlage ist das letzte Kontext-Token
    x_template = @view X_ctx[:, end, 1]
    make_flow_token_from_template!(flow_vec, x_template, x_t, t_idx)
    X = hcat(Float32.(X_ctx), reshape(flow_vec, D_IN, 1, 1))  # (D_IN, ctx+1, 1)
    v = model(X, [t])   # (D_OUT, 1)
    return vec(v)       # (D_OUT,)
end

# Explizites Euler: x_{n+1} = x_n + h * v_theta(x_n, t_n)
function integrate_cfm_euler(model, X_ctx::AbstractArray{<:Real,3},
                             t_idx::Int; steps::Int=8, x0::Union{Nothing,AbstractVector}=nothing,
                             rng::AbstractRNG=Random.default_rng())::Vector{Float32}
    h = 1f0/steps
    x = x0 === nothing ? randn(rng, Float32, D_OUT) : Float32.(x0)
    t = 0f0
    @inbounds for _ in 1:steps
        v = fm_velocity(model, X_ctx, x, t, t_idx)
        x .+= h .* v
        t += h
    end
    return x   # z-normalisierte Vorhersage für den nächsten Messvektor
end

# Midpoint (RK2/Heun): k1 = v(x_n, t_n), x_mid = x_n + 0.5h*k1,
#                       k2 = v(x_mid, t_n+0.5h), x_{n+1} = x_n + h*k2
function integrate_cfm_midpoint(model, X_ctx::AbstractArray{<:Real,3},
                                t_idx::Int; steps::Int=8, x0::Union{Nothing,AbstractVector}=nothing,
                                rng::AbstractRNG=Random.default_rng())::Vector{Float32}
    h = 1f0/steps
    x = x0 === nothing ? randn(rng, Float32, D_OUT) : Float32.(x0)
    t = 0f0
    @inbounds for _ in 1:steps
        k1 = fm_velocity(model, X_ctx, x, t, t_idx)
        xmid = x .+ (0.5f0*h) .* k1
        k2 = fm_velocity(model, X_ctx, xmid, t + 0.5f0*h, t_idx)
        x .+= h .* k2
        t += h
    end
    return x
end