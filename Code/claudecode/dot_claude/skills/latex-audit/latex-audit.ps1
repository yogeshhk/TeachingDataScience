<#
.SYNOPSIS
  Deterministic checks for these LaTeX-teaching repos, so Claude doesn't have to
  re-derive them by reading and reasoning over every file each time.

.DESCRIPTION
  Four modes:
    frames   - count live (uncommented) vs raw \begin{frame} occurrences.
               "Raw" counts the substring anywhere (what a plain grep -c would find,
               including inside %-commented lines). "Live" excludes lines whose
               trimmed text starts with %. Always trust Live when judging deck size.
    inputs   - starting from one or more driver files, transitively walk \input{}/
               \include{} chains (skipping commented-out lines) and report any
               target with no matching .tex on disk in the including file's own
               directory (these repos keep each family folder self-contained, so
               a topic file is always a sibling of the file that \inputs it).
    headings - starting from one or more driver files, walk the same \input/\include
               chain and scan every \frametitle{...} in it for two candidate issues:
               (a) manually-assigned numbering baked into the title text (Exercise 1,
               Lab 2, Step 3, Phase 4, ...) that makes reordering frames painful, and
               (b) the frame title redundantly repeating a word from the driver's own
               \title{...} that a reader already knows from the deck title. BOTH ARE
               CANDIDATE LISTS, NOT VERDICTS -- this is pattern-matching, not judgment.
               Real decks have legitimate exceptions (canonical sequences like
               "Postulate 1..7", official exam section numbers, a frame naming a
               second framework by contrast) -- read the flagged frame before deciding
               to change it, the same way you would with an `inputs` BLOCKED result.
    spacing  - recursively finds every template_cheatsheet.tex file and reports
               whether it has the compact-list fix (enumitem loaded + \setlist{...}
               containing nosep, both uncommented). Flags any copy that's missing it,
               so the fix stays consistent across every family folder / repo that
               carries its own copy of this template.

  All modes skip files under any directory literally named 'backup', '_backup',
  or '_retired' (case-insensitive) -- these repos' own convention for read-only
  archives / dead content that are not meant to be compiled or referenced.

.PARAMETER Mode
  'frames', 'inputs', 'headings', or 'spacing'.

.PARAMETER Path
  A single .tex file, or a directory to scan.
  - frames + directory: reports every .tex file found recursively.
  - inputs/headings + directory: treats every Main_*.tex file found recursively as a
    driver and resolves each one's full chain independently.
  - spacing + file: checks that one file (expected to be a template_cheatsheet.tex).
  - spacing + directory: recurses for every file literally named template_cheatsheet.tex.

.EXAMPLE
  latex-audit.ps1 -Mode frames -Path .\ml_concepts.tex
  latex-audit.ps1 -Mode frames -Path .\LaTeX\
  latex-audit.ps1 -Mode inputs -Path .\Main_Seminar_ML_Intro_Presentation.tex
  latex-audit.ps1 -Mode inputs -Path .\LaTeX\
  latex-audit.ps1 -Mode headings -Path .\Main_Seminar_Qiskit_Presentation.tex
  latex-audit.ps1 -Mode headings -Path .\LaTeX\
  latex-audit.ps1 -Mode spacing -Path .\LaTeX\
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('frames', 'inputs', 'headings', 'spacing')]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [string]$Path
)

function Test-IsExcluded([string]$fullPath) {
    return $fullPath -match '[\\/](backup|_backup|_retired)[\\/]'
}

function Get-FrameCounts([string]$file) {
    $lines = Get-Content -LiteralPath $file
    $raw = 0
    $live = 0
    foreach ($line in $lines) {
        if ($line -match '\\begin\{frame\}') {
            $raw++
            if ($line.TrimStart() -notmatch '^%') { $live++ }
        }
    }
    [PSCustomObject]@{ File = $file; Raw = $raw; Live = $live }
}

function Get-InputTargets([string]$file) {
    $targets = @()
    foreach ($line in Get-Content -LiteralPath $file) {
        if ($line.TrimStart() -match '^%') { continue }
        foreach ($m in [regex]::Matches($line, '\\(input|include)\{([^}]+)\}')) {
            $targets += $m.Groups[2].Value
        }
    }
    return $targets
}

function Resolve-InputChain([string]$startFile) {
    $startFull = (Resolve-Path -LiteralPath $startFile).Path
    $visited = New-Object System.Collections.Generic.HashSet[string]
    $missing = New-Object System.Collections.Generic.List[object]
    $queue = New-Object System.Collections.Generic.Queue[string]
    $queue.Enqueue($startFull)

    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        if ($visited.Contains($current)) { continue }
        $visited.Add($current) | Out-Null
        if (-not (Test-Path -LiteralPath $current)) { continue }

        $dir = [System.IO.Path]::GetDirectoryName($current)
        foreach ($name in (Get-InputTargets $current)) {
            $targetName = if ($name -match '\.tex$') { $name } else { "$name.tex" }
            $candidate = Join-Path $dir $targetName
            if (Test-Path -LiteralPath $candidate) {
                $resolved = (Resolve-Path -LiteralPath $candidate).Path
                if (-not $visited.Contains($resolved)) { $queue.Enqueue($resolved) }
            } else {
                $missing.Add([PSCustomObject]@{
                    Driver   = $startFull
                    From     = $current
                    Target   = $name
                    Expected = $candidate
                })
            }
        }
    }
    [PSCustomObject]@{
        Driver      = $startFull
        FilesWalked = $visited.Count
        Missing     = $missing
        VisitedFiles = ($visited | Sort-Object)
    }
}

# ---------------------------------------------------------------------------
# headings mode
# ---------------------------------------------------------------------------

$HeadingStopWords = @(
    'a','an','the','for','of','and','to','with','non','physicists',
    'quantum','computing','computation','seminar','workshop','overview',
    'cheatsheet','cheat','sheet','reference','presentation','lab',
    'programming','notes','prep','developer','associate','exam',
    'certification','cert','course','course','edx','openedx'
)

$HeadingNumberPattern = '(?i)\b(Exercise|Lab Exercise|Step|Phase|Part|Module|Unit|Task|Chapter|Section|Round|Stage|Postulate)\s*\d+'

function Get-DriverTitle([string]$driverFile) {
    foreach ($line in Get-Content -LiteralPath $driverFile) {
        if ($line.TrimStart() -match '^%') { continue }
        $m = [regex]::Match($line, '\\title(\[[^\]]*\])?\{([^}]*)\}')
        if ($m.Success) { return $m.Groups[2].Value }
    }
    return $null
}

function Get-SubjectPhraseAndWords([string]$titleText) {
    if (-not $titleText) { return $null }
    # Strip LaTeX commands and braces roughly, keep letters/digits/spaces/hyphen/colon.
    $clean = [regex]::Replace($titleText, '\\[a-zA-Z]+', ' ')
    $clean = [regex]::Replace($clean, '[{}]', ' ')

    $phrase = $clean
    if ($clean -match ':') {
        $parts = $clean -split ':'
        $phrase = $parts[$parts.Count - 1].Trim()
    } elseif ($clean -match '(?i)\bfor\b') {
        $phrase = ($clean -split '(?i)\bfor\b')[0].Trim()
    }
    $phrase = $phrase.Trim()

    $words = [regex]::Matches($clean, '[A-Za-z0-9\-]+') |
        ForEach-Object { $_.Value } |
        Where-Object { $_.Length -ge 3 -and ($HeadingStopWords -notcontains $_.ToLower()) } |
        Select-Object -Unique

    [PSCustomObject]@{ Phrase = $phrase; Words = $words }
}

function Get-FrametitlesFromFile([string]$file) {
    $results = @()
    $lineNum = 0
    foreach ($line in Get-Content -LiteralPath $file) {
        $lineNum++
        if ($line.TrimStart() -match '^%') { continue }
        foreach ($m in [regex]::Matches($line, '\\frametitle\{([^}]*)\}')) {
            $results += [PSCustomObject]@{ File = $file; Line = $lineNum; Text = $m.Groups[1].Value }
        }
        foreach ($m in [regex]::Matches($line, '\\begin\{frame\}(\[[^\]]*\])?\{([^}]*)\}')) {
            $results += [PSCustomObject]@{ File = $file; Line = $lineNum; Text = $m.Groups[2].Value }
        }
    }
    return $results
}

function Invoke-HeadingsCheck([string]$driverFile) {
    $chain = Resolve-InputChain $driverFile
    $titleText = Get-DriverTitle $driverFile
    $subject = Get-SubjectPhraseAndWords $titleText

    $allTitles = @()
    foreach ($f in $chain.VisitedFiles) {
        if (Test-IsExcluded $f) { continue }
        $allTitles += Get-FrametitlesFromFile $f
    }

    $numbered = $allTitles | Where-Object { $_.Text -match $HeadingNumberPattern }
    $redundant = @()
    if ($subject -and $subject.Words.Count -gt 0) {
        foreach ($t in $allTitles) {
            $hit = $subject.Words | Where-Object { $t.Text -match "(?i)\b$([regex]::Escape($_))\b" }
            if ($hit) {
                $redundant += [PSCustomObject]@{ File = $t.File; Line = $t.Line; Text = $t.Text; Matched = ($hit -join ', ') }
            }
        }
    }

    "=== $driverFile ==="
    "  title: $titleText"
    if ($subject) { "  subject phrase: '$($subject.Phrase)'  |  candidate words: $($subject.Words -join ', ')" }
    ""
    "  possible reorder-numbering ($($numbered.Count)):"
    foreach ($n in $numbered) { "    $($n.File):$($n.Line): $($n.Text)" }
    ""
    "  possible redundant subject naming ($($redundant.Count)):"
    foreach ($r in $redundant) { "    $($r.File):$($r.Line): $($r.Text)   [matched: $($r.Matched)]" }
    ""
}

# ---------------------------------------------------------------------------
# spacing mode
# ---------------------------------------------------------------------------

function Test-CompactListFix([string]$file) {
    $lines = Get-Content -LiteralPath $file
    $hasPackage = $false
    $hasNosep = $false
    foreach ($line in $lines) {
        if ($line.TrimStart() -match '^%') { continue }
        if ($line -match '\\usepackage(\[[^\]]*\])?\{enumitem\}') { $hasPackage = $true }
        if ($line -match '\\setlist(\[[^\]]*\])?\{[^}]*nosep[^}]*\}') { $hasNosep = $true }
    }
    [PSCustomObject]@{ File = $file; HasPackage = $hasPackage; HasNosep = $hasNosep }
}

# ---------------------------------------------------------------------------

if ($Mode -eq 'frames') {
    $files = if (Test-Path -LiteralPath $Path -PathType Container) {
        Get-ChildItem -LiteralPath $Path -Filter *.tex -Recurse -File |
            Where-Object { -not (Test-IsExcluded $_.FullName) } |
            Select-Object -ExpandProperty FullName
    } else {
        @((Resolve-Path -LiteralPath $Path).Path)
    }

    $results = $files | ForEach-Object { Get-FrameCounts $_ } | Sort-Object File
    $totalRaw = ($results | Measure-Object -Property Raw -Sum).Sum
    $totalLive = ($results | Measure-Object -Property Live -Sum).Sum

    foreach ($r in $results) {
        $flag = if ($r.Raw -ne $r.Live) { "  <- $($r.Raw - $r.Live) commented out" } else { "" }
        "{0,5} live / {1,5} raw   {2}{3}" -f $r.Live, $r.Raw, $r.File, $flag
    }
    ""
    "TOTAL: $totalLive live / $totalRaw raw across $($results.Count) file(s)"
    exit 0
}

if ($Mode -eq 'inputs') {
    $drivers = if (Test-Path -LiteralPath $Path -PathType Container) {
        Get-ChildItem -LiteralPath $Path -Filter 'Main_*.tex' -Recurse -File |
            Where-Object { -not (Test-IsExcluded $_.FullName) } |
            Select-Object -ExpandProperty FullName
    } else {
        @((Resolve-Path -LiteralPath $Path).Path)
    }

    $blockedCount = 0
    foreach ($d in ($drivers | Sort-Object)) {
        $result = Resolve-InputChain $d
        if ($result.Missing.Count -gt 0) {
            $blockedCount++
            "BLOCKED: $($result.Driver)  ($($result.FilesWalked) files walked)"
            foreach ($m in $result.Missing) {
                "    missing '$($m.Target)' (expected $($m.Expected)), referenced from $($m.From)"
            }
        }
    }
    ""
    "$blockedCount of $($drivers.Count) driver(s) blocked by an unresolved \input/\include."
    exit 0
}

if ($Mode -eq 'headings') {
    $drivers = if (Test-Path -LiteralPath $Path -PathType Container) {
        Get-ChildItem -LiteralPath $Path -Filter 'Main_*.tex' -Recurse -File |
            Where-Object { -not (Test-IsExcluded $_.FullName) } |
            Select-Object -ExpandProperty FullName
    } else {
        @((Resolve-Path -LiteralPath $Path).Path)
    }

    foreach ($d in ($drivers | Sort-Object)) {
        Invoke-HeadingsCheck $d
    }
    exit 0
}

if ($Mode -eq 'spacing') {
    $files = if (Test-Path -LiteralPath $Path -PathType Container) {
        Get-ChildItem -LiteralPath $Path -Filter 'template_cheatsheet.tex' -Recurse -File |
            Where-Object { -not (Test-IsExcluded $_.FullName) } |
            Select-Object -ExpandProperty FullName
    } else {
        @((Resolve-Path -LiteralPath $Path).Path)
    }

    $missingCount = 0
    foreach ($f in ($files | Sort-Object)) {
        $r = Test-CompactListFix $f
        if ($r.HasPackage -and $r.HasNosep) {
            "OK       $($r.File)"
        } else {
            $missingCount++
            $why = @()
            if (-not $r.HasPackage) { $why += 'enumitem not loaded (uncommented)' }
            if (-not $r.HasNosep) { $why += 'no \setlist{...nosep...} found (uncommented)' }
            "MISSING  $($r.File)   [$($why -join '; ')]"
        }
    }
    ""
    "$missingCount of $($files.Count) template_cheatsheet.tex file(s) missing the compact-list fix."
    exit 0
}
