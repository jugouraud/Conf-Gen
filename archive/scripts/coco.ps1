param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $CocoArguments
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$distribution = if ($env:CONF_GEN_WSL_DISTRO) { $env:CONF_GEN_WSL_DISTRO } else { "Ubuntu" }
$distributions = & wsl.exe --list --quiet

if ($LASTEXITCODE -ne 0 -or -not $distributions) {
    throw "Google Colab CLI requires Linux or macOS. Install WSL first with 'wsl --install -d Ubuntu', restart Windows, and rerun this command."
}

& wsl.exe -d $distribution --cd $repositoryRoot bash -l ./scripts/coco.sh @CocoArguments
exit $LASTEXITCODE
