$ErrorActionPreference = 'Stop'

# [Console]::In decodes stdin using the console's legacy OEM/ANSI codepage, not UTF-8,
# which silently mangles multi-byte characters like the em-dash (U+2014, E2 80 94) before
# they ever reach the regex below. Read raw bytes via a StreamReader with UTF-8 forced
# instead, so the harness's UTF-8-encoded JSON payload is decoded correctly.
$reader = New-Object System.IO.StreamReader([Console]::OpenStandardInput(), [System.Text.Encoding]::UTF8)
$stdin = $reader.ReadToEnd()
try {
    $payload = $stdin | ConvertFrom-Json
} catch {
    exit 0
}

$toolName = $payload.tool_name
if ($toolName -ne 'Edit' -and $toolName -ne 'Write') { exit 0 }

$filePath = $payload.tool_input.file_path
if (-not $filePath) { exit 0 }

$ext = [System.IO.Path]::GetExtension($filePath).ToLower()
if ($ext -ne '.tex' -and $ext -ne '.md') { exit 0 }

$fileName = [System.IO.Path]::GetFileName($filePath)
if ($fileName -eq 'CLAUDE.md') { exit 0 }

if ($toolName -eq 'Write') {
    $text = $payload.tool_input.content
} else {
    $text = $payload.tool_input.new_string
}
if ($null -eq $text -or $text -eq '') { exit 0 }

# Text used for pattern matching: strip content this repo's rules exempt.
$scanText = $text
if ($ext -eq '.tex') {
    $scanText = ($scanText -split "`n" | Where-Object { $_ -notmatch '^\s*%' }) -join "`n"
    $scanText = [regex]::Replace($scanText, '(?s)\\begin\{(lstlisting|verbatim)\}.*?\\end\{\1\}', '')
}

$lines = $text -split "`n"

function Find-Line($pattern) {
    for ($i = 0; $i -lt $lines.Length; $i++) {
        if ($lines[$i] -match $pattern) {
            return @{ num = $i + 1; text = $lines[$i].Trim() }
        }
    }
    return $null
}

$emdash = [char]0x2014
if ($scanText.IndexOf($emdash) -ge 0) {
    $hit = Find-Line([regex]::Escape($emdash))
    $where = if ($hit) { " near line $($hit.num): `"$($hit.text)`"" } else { "" }
    $reason = "Blocked: literal em-dash (U+2014) found$where. This repo's CLAUDE.md bans em-dashes in prose. Rephrase using a colon, semicolon, comma, or hyphen (for quote attribution) -- pick whichever reads naturally for this sentence, do not do a mechanical find-replace."
    $output = @{
        hookSpecificOutput = @{
            hookEventName            = 'PreToolUse'
            permissionDecision       = 'deny'
            permissionDecisionReason = $reason
        }
    }
    $output | ConvertTo-Json -Depth 5 -Compress
    exit 0
}

if ($scanText -match '--') {
    $hit = Find-Line('--')
    $where = if ($hit) { " near line $($hit.num): `"$($hit.text)`"" } else { "" }
    $reason = "Warning: ASCII '--' or '---' found$where (outside %-comments and lstlisting/verbatim). Could be a fake em/en-dash in prose (this repo's convention bans that) OR legitimate syntax (e.g. TikZ 'a -- b', a CLI flag). Check the context before proceeding; not auto-blocked."
    $output = @{
        hookSpecificOutput = @{
            hookEventName      = 'PreToolUse'
            permissionDecision = 'allow'
            additionalContext  = $reason
        }
        systemMessage = $reason
    }
    $output | ConvertTo-Json -Depth 5 -Compress
    exit 0
}

exit 0
