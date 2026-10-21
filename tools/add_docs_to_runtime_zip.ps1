# Adds the four GitHub Markdown references to an existing validated runtime ZIP.
# Usage from project root: .\tools\add_docs_to_runtime_zip.ps1
# Creates a backup before modifying the original ZIP. Does not alter source code or weights.
param(
    [string]$ZipPath = 'C:\responsible-ai-healthcare-llms-final.zip'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$ZipPath = [System.IO.Path]::GetFullPath($ZipPath)
$docs = @(
    'docs/DATASET_MANIFEST.md',
    'docs/APPLICATION_ARCHITECTURE.md',
    'docs/DEPLOYMENT_GUIDE.md',
    'docs/DEMO_USER_MANUAL.md'
)

if (-not (Test-Path -LiteralPath $ZipPath -PathType Leaf)) { throw "Missing ZIP: $ZipPath" }
foreach ($rel in $docs) {
    $path = Join-Path $Root ($rel.Replace('/', '\'))
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $path" }
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$backup = [System.IO.Path]::ChangeExtension($ZipPath, '.before_docs.zip')
if (Test-Path -LiteralPath $backup) { throw "Backup already exists, refusing overwrite: $backup" }
Copy-Item -LiteralPath $ZipPath -Destination $backup
Write-Host "Backup created: $backup"

$zip = [System.IO.Compression.ZipFile]::Open($ZipPath, [System.IO.Compression.ZipArchiveMode]::Update)
try {
    $codeFile = $zip.Entries | Where-Object {
        $_.FullName.Replace('\', '/').EndsWith('src/demo/app.py')
    } | Select-Object -First 1
    if (-not $codeFile) { throw 'Cannot identify runtime ZIP root: src/demo/app.py not found.' }
    $normal = $codeFile.FullName.Replace('\', '/')
    $prefix = $normal.Substring(0, $normal.Length - 'src/demo/app.py'.Length)
    Write-Host "Using ZIP folder prefix: '$prefix'"

    foreach ($rel in $docs) {
        $target = $prefix + $rel
        $old = $zip.Entries | Where-Object { $_.FullName.Replace('\', '/') -eq $target } | Select-Object -First 1
        if ($old) { $old.Delete() }
        $source = Join-Path $Root ($rel.Replace('/', '\'))
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, $source, $target, [System.IO.Compression.CompressionLevel]::Optimal
        )
        Write-Host "Added: $target"
    }
}
finally { $zip.Dispose() }

$verify = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
try {
    foreach ($rel in $docs) {
        if (-not @($verify.Entries | Where-Object { $_.FullName.Replace('\', '/').EndsWith($rel) }).Count) {
            throw "ZIP check failed: $rel"
        }
    }
    Write-Host 'PASS: All four documents found in updated runtime ZIP.'
}
finally { $verify.Dispose() }
