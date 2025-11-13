using CSV, DataFrames, Statistics, Printf



# exakt wie in deinem finalen Scraper:
const NUMERIC_COLS = [
    "wind_mean_ms","wind_max_ms","wind_min_ms",
    "rpm_mean","rpm_max","rpm_min",
    "power_mean_kw","power_max_kw","power_min_kw",
    "power_avail_wind_mean_kw",
    "power_avail_tech_mean_kw",
    "power_avail_force_maj_mean_kw",
    "power_avail_ext_mean_kw",
]

const SEASONS = ("Winter","Fruehling","Sommer","Herbst")

# -------------- Welford -----------------
mutable struct Welford
    n::Int64
    mean::Float64
    m2::Float64
end
Welford() = Welford(0, 0.0, 0.0)

@inline function update!(w::Welford, x::Float64)
    w.n += 1
    δ   = x - w.mean
    w.mean += δ / w.n
    δ2  = x - w.mean
    w.m2 += δ * δ2

    if isnan(w.mean) || isnan(w.m2)
        @show x, w.n, w.mean, w.m2
        error("A")
    end

    return w
end

@inline function finalize(w::Welford)
    if w.n <= 1
        return (μ = w.mean, σ = 0.0, n = w.n)
    else
        return (μ = w.mean, σ = sqrt(w.m2 / (w.n - 1)), n = w.n)
    end
end

# (season, serial, feature) -> accumulator
ACC = Dict{Tuple{String,String,String}, Welford}()

@inline function acc_ref(season::String, serial::String, feat::String)
    get!(ACC, (season, serial, feat)) do
        Welford()
    end
end

# -------------- Helpers -----------------
# alle CSV Pfade unter HK_blocks/Season/Serial/
function list_csvs(base::AbstractString)
    paths = String[]
    for s in SEASONS
        sdir = joinpath(base, s)
        isdir(sdir) || continue
        for serial in readdir(sdir)
            ssdir = joinpath(sdir, serial)
            isdir(ssdir) || continue
            for f in readdir(ssdir; join=true)
                endswith(lowercase(f), ".csv") && push!(paths, f)
            end
        end
    end
    return paths
end

# Seriennummer aus Pfad extrahieren (…/Season/Serial/file.csv)
function serial_from_path(path::AbstractString)
    parts = splitpath(path)
    # .../[BASE]/Season/Serial/File -> vorletztes Segment
    return parts[end-1]
end

# Season aus Pfad extrahieren
function season_from_path(path::AbstractString)
    parts = splitpath(path)
    return parts[end-2]
end

# -------------- Main --------------------
function compute_log_stats(BASE_DIR  = "HK_blocks", out_json   = "HK_blocks/log_stats_by_serial_year.json")
    global ACC = Dict{Tuple{String,String,String}, Welford}()
    files = list_csvs(BASE_DIR)
    println("Gefundene CSV-Blöcke: ", length(files))

    num_files = 0
    num_rows  = 0
    num_vals  = 0

    global test = []
    global nn = 0

    for p in files
        season = season_from_path(p)
        serial = serial_from_path(p)

        # nur unsere 4 Season-Ordner berücksichtigen
        season in SEASONS || continue

        # unsere vorbereiteten CSVs haben Dezimalpunkt; kein decimal/groupmark nötig
        global df = CSV.read(p, DataFrame; normalizenames=false)

        # sanity: vorhandene numerische Spalten
        cols_here = intersect(NUMERIC_COLS, String.(names(df)))
        isempty(cols_here) && continue

        num_files += 1
        num_rows  += nrow(df)

        for i in 1:nrow(df)
            all_zeros = true

            for j in NUMERIC_COLS
                if df[i,j] > 0
                    all_zeros = false
                end
            end

            if all_zeros
                nn += 1
                @show p, i
            end
        end

        # pro Spalte Werte streamen
        for c in cols_here
            col = df[!, c]

            @inbounds for i in 1:length(col)
                x = col[i]

                if x !== missing
                    if ((c == "wind_min_ms" || c == "rpm_min") && x > 70)
                        nn += 1
                        @show p, i
                    end

                    isnan(log(x)) && @show x

                    update!(acc_ref(season, serial, c), Float64(log(x)))
                    num_vals += 1
                end
            end
        end
    end

    println(@sprintf("Dateien: %d | Zeilen (gesamt): %d | Werte (gezählt): %d",
                     num_files, num_rows, num_vals))

    # ACC -> DataFrame
    rows = Vector{NamedTuple{(:season,:serial,:feature,:mu,:sigma,:n),
                             Tuple{String,String,String,Float64,Float64,Int64}}}()
    for ((s, sn, f), w) in ACC
        stats = finalize(w)
        push!(rows, (s, sn, f, stats.μ, stats.σ, stats.n))
    end
    global outdf = DataFrame(rows)
    sort!(outdf, [:season, :serial, :feature])


    open(out_json, "w") do io
        #JSON3.write(io, out; allow_inf=true, indent=2)
        JSON3.pretty(io, JSON3.write(outdf))
    end
    println("Wrote ", out_json)

    # CSV.write(OUT_CSV, outdf)
    # println("Geschrieben: ", OUT_CSV)

    # optional: schnelle Ausreißer-Sicht (sehr grob)
    # show(first(sort(outdf, :sigma, rev=true), 20), allcols=true, truncate=120)
end

