param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $EmbeddingArguments
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$distributions = & wsl.exe --list --quiet

if ($LASTEXITCODE -ne 0 -or -not $distributions) {
    throw "Google Colab CLI requires Linux or macOS. Install WSL first with 'wsl --install -d Ubuntu', restart Windows, and rerun this command."
}

& wsl.exe --cd $repositoryRoot python3 ./scripts/colab_embeddings.py @EmbeddingArguments
exit $LASTEXITCODE
