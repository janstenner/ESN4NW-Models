using CSV, DataFrames, Dates

# ---- Helper: meteorologische Season -> OneHot(4) ----
@inline function season_onehot(dt::DateTime)::NTuple{4,Float32}
    m = month(dt)
    # Winter=1 (Dez/Jan/Feb), Frühling=2 (Mär–Mai), Sommer=3 (Jun–Aug), Herbst=4 (Sep–Nov)
    s = (m in (12,1,2)) ? 1 : (m in (3,4,5) ? 2 : (m in (6,7,8) ? 3 : 4))
    return (Float32(s==1), Float32(s==2), Float32(s==3), Float32(s==4))
end

# ---- Helper: Zeit als sin/cos (Minute des Tages) ----
@inline function time_sin_cos(dt::DateTime)::NTuple{2,Float32}
    min_of_day = hour(dt)*60 + minute(dt)
    θ = 2f0 * π * Float32(min_of_day) / 1440f0
    return (sin(θ), cos(θ))
end

# ---- Wähle die kontinuierlichen Kanäle in fester Reihenfolge ----
const ORDERED_BASE = [
    "wind_mean_ms","wind_max_ms","wind_min_ms",
    "rpm_mean","rpm_max","rpm_min",
    "power_mean_kw","power_max_kw","power_min_kw",
    "power_avail_wind_mean_kw",
    "power_avail_tech_mean_kw",
    "power_avail_force_maj_mean_kw",
    "power_avail_ext_mean_kw",
]


function pick_numeric_columns(df::DataFrame)::Vector{String}
    names_set = Set(String.(names(df)))
    if all(in(names_set), ORDERED_BASE)
        return ORDERED_BASE
    else
        missing = setdiff(ORDERED_BASE, collect(names_set))
        error("Erwartete Spalten fehlen: $(missing). Verfügbare: $(names(df))")
    end
end

# ---- Hauptlogik: eine CSV lesen, erste Zeile in Embedding umwandeln ----
function main(path::AbstractString)
    # CSV mit deutschem Dezimal-Komma robust einlesen (Punkt als Tausendertrenner wird ignoriert)
    df = CSV.read(path, DataFrame; normalizenames=false)
    # Wir erwarten vorbereitete/später umbenannte Spalten (snake_case), inkl. :time, :serial, :alias
    @assert "time" ∈ names(df) "Spalte 'time' fehlt. Bitte vorbereitete CSV (Scraper-Output) verwenden."

    # erste Datenzeile nach Header
    row = df[1, :]

    # 1) Zeitfeatures
    dt  = row[:time] isa DateTime ? row[:time] : DateTime(row[:time])
    s4  = season_onehot(dt)              # (4,)
    sc  = time_sin_cos(dt)               # (2,)

    # 2) Kontinuierliche Kanäle (14)
    cols = pick_numeric_columns(df)
    vals = Float32[
        (row[Symbol(c)] isa Missing) ? NaN32 : Float32(row[Symbol(c)])
        for c in cols
    ]

    # 3) Finales Embedding: season(4) ⊕  ⊕ 14-Kanäle  => Länge 20 (bei 14 Kanälen)
    emb = Float32[v for v in s4]; append!(emb, (Float32(sc[1]), Float32(sc[2])))
    append!(emb, vals)

    println("Embedding length: ", length(emb))
    println("Embedding: ", emb)
    println("Columns used: ", cols)

    emb
end


emb = main("HK_blocks/Fruehling/920696/WEA_10 Minuten_6556_2021-01-01_2021-12-31_Fruehling_920696_block00009.csv")

@show emb
