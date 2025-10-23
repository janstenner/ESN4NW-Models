using CSV, DataFrames, Dates, Printf
using StringEncodings

# ----------------------------- Konfiguration -----------------------------
const IN_DIR  = "Huser_Klee_complete/"
const OUT_DIR = "HK_blocks"
const STEP = Minute(10)

const MIN_BLOCK_ROWS = 2   # vorher 6 – zum Testen runtersetzen
const DEBUG = true

# ----------------------------- Utilities -----------------------------

using Dates

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

    # gewünschte Originalnamen (deine Liste)
    KEEP_COLS_ORIG = [
        "Zeit",
        "Wind Ø [m/s]","Wind max. [m/s]","Wind min. [m/s]",
        "Drehzahl Ø [1/min]","Drehzahl max. [1/min]","Drehzahl min. [1/min]",
        "Leistung Ø [kW]","Leistung max. [kW]","Leistung min. [kW]",
        "Leistung Verfügb","Wind Ø [kW]","Leistung Verfügb. techn. Ø [kW]",
        "Leistung Verfügb. force maj. Ø [kW]",
        "Leistung Verfügb. ext. Ø [kW]",
        "Anlage","Seriennr.","Alias"
    ]

    RENAME_MAP = Dict(
        "Zeit"=>"time",
        "Wind Ø [m/s]"=>"wind_mean_ms","Wind max. [m/s]"=>"wind_max_ms","Wind min. [m/s]"=>"wind_min_ms",
        "Drehzahl Ø [1/min]"=>"rpm_mean","Drehzahl max. [1/min]"=>"rpm_max","Drehzahl min. [1/min]"=>"rpm_min",
        "Leistung Ø [kW]"=>"power_mean_kw","Leistung max. [kW]"=>"power_max_kw","Leistung min. [kW]"=>"power_min_kw",
        "Leistung Verfügb"=>"power_avail","Wind Ø [kW]"=>"wind_power_mean_kw",
        "Leistung Verfügb. techn. Ø [kW]"=>"power_avail_tech_mean_kw",
        "Leistung Verfügb. force maj. Ø [kW]"=>"power_avail_force_maj_mean_kw",
        "Leistung Verfügb. ext. Ø [kW]"=>"power_avail_ext_mean_kw",
        "Anlage"=>"plant","Seriennr."=>"serial","Alias"=>"alias"
    )

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
function split_into_blocks(df::DataFrame)
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
        gap = (t - last_t != STEP)
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

function save_block_csv(block::DataFrame; outdir::String, base::String, idx::Int)
    if nrow(block) < MIN_BLOCK_ROWS
        return nothing
    end
    
    s = season(block[1, "time"])
    season_dir = joinpath(outdir, s)
    isdir(season_dir) || mkpath(season_dir)

    # kurze Blöcke (< ~1h) ignorieren
    if nrow(block) < 6
        return nothing
    end

    outpath = joinpath(season_dir, @sprintf("%s_%s_block%05d.csv", base, s, idx))
    # CSV.write schreibt numerische Werte mit Dezimalpunkt; DataFrames -> CSV ist Standard. :contentReference[oaicite:2]{index=2}
    CSV.write(outpath, block)
    return outpath
end

# ----------------------------- Main -----------------------------
isdir(OUT_DIR) || mkpath(OUT_DIR)
csv_files = filter(f -> endswith(lowercase(f), ".csv"), readdir(IN_DIR; join=true))
println("Gefundene CSVs: ", length(csv_files))

total_blocks = 0
for (fi, path) in enumerate(csv_files)
    global total_blocks
    println("Lese: $path")
    df = read_csv_robust(path)           # Dezimal-Komma ✓ :contentReference[oaicite:3]{index=3}
    sub = select_sort_and_rename(df)     # umbenennen ✓
    blocks = split_into_blocks(sub)
    base = splitext(basename(path))[1]

    if DEBUG
        println("  -> erkannte Spalten: ", names(sub))
        println("  -> Zeilen: ", nrow(sub))
    end

    blocks = split_into_blocks(sub)

    if DEBUG
        println("  -> Blöcke gesamt: ", length(blocks))
        if !isempty(blocks)
            println("  -> Block 1 Länge: ", nrow(blocks[1]))
        end
    end

    saved = 0
    for (i, blk) in enumerate(blocks)
        out = save_block_csv(blk; outdir=OUT_DIR, base=base, idx=i)
        saved += isnothing(out) ? 0 : 1
    end
    total_blocks += saved
    println(" -> gespeichert: $(saved) Blöcke")
end



println("Fertig. Gesamt gespeicherte Blöcke: $total_blocks")