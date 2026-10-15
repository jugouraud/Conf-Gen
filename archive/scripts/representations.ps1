param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ComparisonArguments
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$distribution = if ($env:CONF_GEN_WSL_DISTRO) { $env:CONF_GEN_WSL_DISTRO } else { "Ubuntu" }
$distributions = & wsl.exe --list --quiet
if ($LASTEXITCODE -ne 0 -or -not $distributions) {
    throw "Google Colab CLI requires WSL. Run this command from a machine where WSL is available."
}
& wsl.exe -d $distribution --cd $repositoryRoot bash -l ./scripts/representations.sh @ComparisonArguments
exit $LASTEXITCODE
