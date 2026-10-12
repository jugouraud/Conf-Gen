param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $GpuArguments
)
$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$distribution = if ($env:CONF_GEN_WSL_DISTRO) { $env:CONF_GEN_WSL_DISTRO } else { "Ubuntu" }
$converted = [System.Collections.Generic.List[string]]::new()

for ($i = 0; $i -lt $GpuArguments.Count; $i++) {
    $argument = $GpuArguments[$i]
    if ($argument -eq "--upload") {
        if ($i + 2 -ge $GpuArguments.Count) { throw "--upload requires LOCAL and REMOTE" }
        $local = (Resolve-Path -LiteralPath $GpuArguments[$i + 1]).Path
        $remote = $GpuArguments[$i + 2]
        $wslLocal = (& wsl.exe -d $distribution -- wslpath -a $local).Trim()
        if ($LASTEXITCODE -ne 0) { throw "Could not convert upload path: $local" }
        $converted.Add("--upload")
        $converted.Add($wslLocal)
        $converted.Add($remote)
        $i += 2
    } elseif ($argument -eq "--download") {
        if ($i + 2 -ge $GpuArguments.Count) { throw "--download requires REMOTE and LOCAL" }
        $remote = $GpuArguments[$i + 1]
        $local = [IO.Path]::GetFullPath($GpuArguments[$i + 2])
        $wslLocal = (& wsl.exe -d $distribution -- wslpath -a $local).Trim()
        if ($LASTEXITCODE -ne 0) { throw "Could not convert download path: $local" }
        $converted.Add("--download")
        $converted.Add($remote)
        $converted.Add($wslLocal)
        $i += 2
    } else {
        $converted.Add($argument)
    }
}

& wsl.exe -d $distribution --cd $repositoryRoot bash -l ./scripts/gpu.sh @converted
exit $LASTEXITCODE
