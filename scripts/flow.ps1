param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $FlowArguments
)
$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$distribution = if ($env:CONF_GEN_WSL_DISTRO) { $env:CONF_GEN_WSL_DISTRO } else { "Ubuntu" }
$distributions = & wsl.exe --list --quiet
if ($LASTEXITCODE -ne 0 -or -not $distributions) {
    throw "Google Colab CLI requires WSL. Install WSL Ubuntu and rerun."
}
& wsl.exe -d $distribution --cd $repositoryRoot bash -l ./scripts/flow.sh @FlowArguments
exit $LASTEXITCODE
