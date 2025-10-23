using CSV, DataFrames, Dates, Printf
using StringEncodings
using Dates

# ----------------------------- Konfiguration -----------------------------
const IN_DIR  = "HK_complete/"
const OUT_DIR = "HK_blocks"
const STEP = Minute(10)

const MIN_BLOCK_ROWS = 100
const DEBUG = true

const STEP_MS = 10 * 60 * 1000      # 10 Minuten in Millisekunden
const TOL_MS  = 5 * 1000            # Toleranzfenster 5s (adjustierbar)


const KEEP_COLS_ORIG = [
    "Zeit",
    "Wind Ø [m/s]","Wind max. [m/s]","Wind min. [m/s]",
    "Drehzahl Ø [1/min]","Drehzahl max. [1/min]","Drehzahl min. [1/min]",
    "Leistung Ø [kW]","Leistung max. [kW]","Leistung min. [kW]",
    "Leistung Verfügb. Wind Ø [kW]",
    "Leistung Verfügb. techn. Ø [kW]",
    "Leistung Verfügb. force maj. Ø [kW]",
    "Leistung Verfügb. ext. Ø [kW]",
    "Anlage","Seriennr.","Alias"
]

const RENAME_MAP = Dict(
    "Zeit" => "time",
    "Wind Ø [m/s]" => "wind_mean_ms",
    "Wind max. [m/s]" => "wind_max_ms",
    "Wind min. [m/s]" => "wind_min_ms",
    "Drehzahl Ø [1/min]" => "rpm_mean",
    "Drehzahl max. [1/min]" => "rpm_max",
    "Drehzahl min. [1/min]" => "rpm_min",
    "Leistung Ø [kW]" => "power_mean_kw",
    "Leistung max. [kW]" => "power_max_kw",
    "Leistung min. [kW]" => "power_min_kw",
    "Leistung Verfügb. Wind Ø [kW]" => "power_avail_wind_mean_kw",
    "Leistung Verfügb. techn. Ø [kW]" => "power_avail_tech_mean_kw",
    "Leistung Verfügb. force maj. Ø [kW]" => "power_avail_force_maj_mean_kw",
    "Leistung Verfügb. ext. Ø [kW]" => "power_avail_ext_mean_kw",

    "Anlage" => "plant",
    "Seriennr." => "serial",
    "Alias" => "alias",
)

const REQUIRED_COLS = [
    "time",
    "wind_mean_ms","wind_max_ms","wind_min_ms",
    "rpm_mean","rpm_max","rpm_min",
    "power_mean_kw","power_max_kw","power_min_kw",
    "power_avail_wind_mean_kw",
    "power_avail_tech_mean_kw",
    "power_avail_force_maj_mean_kw",
    "power_avail_ext_mean_kw",
]

# ----------------------------- Utilities -----------------------------

# Ersetzt alle "exotischen" Whitespaces (NBSP, schmale NBSP, etc.) durch normale Spaces
# und schneidet vorne/hinten ab.
clean_time_str(s) = begin
    t = String(s)
    # häufige Kandidaten normalisieren:
    t = replace(t, '\u00A0' => ' ', '\u2007' => ' ', '\u202F' => ' ')
    # alle Unicode-Whitespace zu Space falten (Fallback):
    t = join(c == ' ' || !isspace(c) ? string(c) : " " for c in t)
    strip(t)
end

# Seriennummer für Pfade/Dateinamen säubern
sanitize_label(s) = replace(string(s), r"[^\w\.\-]+" => "_")

# Robustes Zeit-Parsing mit mehreren Formatversuchen
function parse_time_any(x)
    s = clean_time_str(x)
    # typische Varianten: ohne Sekunden / mit Sekunden
    for df in (dateformat"dd.mm.yyyy HH:MM",
               dateformat"dd.mm.yyyy HH:MM:SS")
        try
            return DateTime(s, df)
        catch
        end
    end
    error("Zeitformat unbekannt (nach Cleaning): $s")
end


season(dt::DateTime) = begin
    m = month(dt)
    m in (12,1,2)  ? "Winter"   :
    m in (3,4,5)   ? "Fruehling" :
    m in (6,7,8)   ? "Sommer"   : "Herbst"
end

function parse_time(x)
    try
        return DateTime(x)
    catch
    end
    try
        return DateTime(x, dateformat"dd.mm.yyyy HH:MM")
    catch
        error("Zeitformat unbekannt: $x")
    end
end

# robustes Einlesen mit Dezimal-Komma; Fallback auf ; als Trennzeichen
function read_csv_robust(path::AbstractString)
    # 1) Standard: UTF-8, Dezimal-Komma, evtl. Delimiter ';'
    for (dlm, hdr) in ((nothing, nothing), (';', nothing))
        try
            df = CSV.read(path, DataFrame; normalizenames=false, decimal=',', groupmark='.', delim=dlm)
            # wenn "Zeit" nicht erkannt wurde, versuche Header in Zeile 2
            if !any(canon.(String.(names(df))) .== canon("Zeit"))
                df = CSV.read(path, DataFrame; normalizenames=false, decimal=',', groupmark='.', delim=dlm, header=2)
            end
            return df
        catch
        end
    end
    # 2) Fallback: Non-UTF-8 via StringEncodings (ISO-8859-1/CP1252 sind häufig)
    try
        io = open(path, enc"ISO-8859-1")
        df = CSV.read(io, DataFrame; normalizenames=false, decimal=',', groupmark='.')
        close(io)
        return df
    catch
        # letzter Versuch mit ; + Latin-1
        try
            io = open(path, enc"ISO-8859-1")
            df = CSV.read(io, DataFrame; normalizenames=false, decimal=',', groupmark='.', delim=';')
            close(io)
            return df
        catch e
            error("CSV-Read fehlgeschlagen für $path — prüfe Encoding/Struktur. Ursprungsfehler: $(e)")
        end
    end
end

# -> wählt relevante Spalten, parst Zeit, sortiert, und benennt um
using Unicode

# Kanonische Normalisierung für Header (BOM, NBSP, Umlaute, Ø, Punkte/Slashes)
canon(s::AbstractString) = begin
    t = replace(String(s),
        '\ufeff' => "", '\u00A0' => ' ', '\u2007' => ' ', '\u202F' => ' ')
    t = lowercase(Unicode.normalize(t, :NFKD))              # diakritika lösen
    t = replace(t, r"[\p{Mn}]" => "")                       # diakritika weg
    t = replace(t, "ä"=>"ae","ö"=>"oe","ü"=>"ue","ß"=>"ss","ø"=>"o")
    t = replace(t, r"\s+" => " ")
    t = replace(t, r"[^\w]+" => "_")
    strip(t, '_')
end

function select_sort_and_rename(df::DataFrame)::DataFrame
    cols = String.(names(df))
    # Map: kanonisierter -> originaler Header
    cmap = Dict(canon(c) => c for c in cols)

    # finde in df die jeweils "entsprechende" Spalte per kanonischem Namen
    found_pairs = [(cmap[canon(orig)], RENAME_MAP[orig]) for orig in KEEP_COLS_ORIG if haskey(cmap, canon(orig))]

    # harte Prüfung auf Zeitspalte
    if !any(p->p[2]=="time", found_pairs)
        # Spezialfall: Header evtl. in Zeile 2 -> Caller soll read_csv_robust neu lesen (B) 
        error("Zeitspalte nicht gefunden (auch nach Header-Normalisierung).")
    end

    keep_actual = first.(found_pairs)            # originale Header im df
    sub = copy(df[!, keep_actual])               # Spalten ziehen
    rename!(sub, Dict(found_pairs))              # in snake_case umbenennen

    # Zeit parsen & sortieren (robust)
    sub[!, "time"] = [parse_time_any(z) for z in sub[!, "time"]]
    sort!(sub, "time")
    return sub
end

# Schneidet in Blöcke, wenn 10-Min-Gap ODER Seasonwechsel
function split_into_blocks(df_in::AbstractDataFrame)
    # Materialisieren (SubDataFrame → DataFrame),
    # damit alle Operationen/Views später "echt" sind:
    df = DataFrame(df_in)

    @assert "time" ∈ names(df)
    T = nrow(df)
    if T == 0; return DataFrame[]; end
    blocks = DataFrame[]
    start_idx = 1
    last_t = df[start_idx, "time"]
    last_s = season(last_t)
    for i in 2:T
        t = df[i, "time"]
        s = season(t)

        diff_ms = Dates.value(t - last_t)    # Millisekunden-Differenz
        gap = abs(diff_ms - STEP_MS) > TOL_MS

        if diff_ms == 0
            last_t = t; last_s = s
            continue
        end



        #gap = (t - last_t != STEP)
        s_change = (s != last_s)
        if gap || s_change
            push!(blocks, DataFrame(@view df[start_idx:(i-1), : ]))
            start_idx = i
        end
        last_t = t
        last_s = s
    end
    push!(blocks, DataFrame(@view df[start_idx:end, : ]))
    return blocks
end

function save_block_csv(block::DataFrame; outdir::String, base::String, idx::Int, serial_label::AbstractString)
    if nrow(block) < MIN_BLOCK_ROWS
        return nothing
    end

    missing_cols = [c for c in REQUIRED_COLS if !(c in names(block))]
    if !isempty(missing_cols)
        DEBUG && @info "Block übersprungen – fehlende Spalten" missing_cols
        return nothing
    end

    s = season(block[1, "time"])
    season_dir = joinpath(outdir, s)

    # Unterordner je Seriennummer
    serial_dir = joinpath(season_dir, sanitize_label(serial_label))
    isdir(serial_dir) || mkpath(serial_dir)

    outpath = joinpath(
        serial_dir,
        @sprintf("%s_%s_%s_block%05d.csv", base, s, sanitize_label(serial_label), idx)
    )

    CSV.write(outpath, block)  # CSV.write lt. Doku. :contentReference[oaicite:1]{index=1}
    return outpath
end

# ----------------------------- Main -----------------------------
isdir(OUT_DIR) || mkpath(OUT_DIR)
csv_files = filter(f -> endswith(lowercase(f), ".csv"), readdir(IN_DIR; join=true))
println("Gefundene CSVs: ", length(csv_files))

total_blocks = 0
for (fi, path) in enumerate(csv_files)
    global total_blocks
    global gi, saved_total_this_file
    global sub
    global gdf

    println("Lese: $path")
    base = splitext(basename(path))[1]

    df = read_csv_robust(path)           # Dezimal-Komma ✓ :contentReference[oaicite:3]{index=3}
    sub = select_sort_and_rename(df)     # umbenennen ✓  :contentReference[oaicite:3]{index=3}

    if DEBUG
        println("  -> erkannte Spalten: ", names(sub))
        println("  -> Zeilen: ", nrow(sub))
    end

    # --- NEU: pro Seriennummer gruppieren ---
    if !("serial" in names(sub))
        @warn "Spalte 'serial' fehlt – Datei wird übersprungen."
        continue
    end

    gdf = groupby(sub, "serial")  # DataFrames.groupby nach Doku. :contentReference[oaicite:4]{index=4}

    saved_total_this_file = 0
    gi = 0
    for g in gdf
        

        gi += 1
        serial_val = g[1, "serial"]    # jede Gruppe hat eine Seriennr.

        blocks = split_into_blocks(g)  # dein bestehender Split in kohärente 10-Min-Blöcke :contentReference[oaicite:5]{index=5}

        if DEBUG
            println("  -> Seriennr=$serial_val: Blöcke=", length(blocks),
                    ", erste Blocklänge=", isempty(blocks) ? 0 : nrow(blocks[1]))
        end

        saved = 0
        for (i, blk) in enumerate(blocks)
            out = save_block_csv(blk; outdir=OUT_DIR, base=base, idx=i, serial_label=string(serial_val))
            saved += isnothing(out) ? 0 : 1
        end
        saved_total_this_file += saved
    end

    total_blocks += saved_total_this_file
    println(" -> gespeichert: $(saved_total_this_file) Blöcke")
end



println("Fertig. Gesamt gespeicherte Blöcke: $total_blocks")