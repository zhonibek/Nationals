param([Parameter(Mandatory=$true)][string]$Directory)
$ErrorActionPreference='Stop'
$folder=(Resolve-Path -LiteralPath $Directory).Path
$data=Get-Content -LiteralPath (Join-Path $folder 'pitch-data.json') -Raw | ConvertFrom-Json
Add-Type -AssemblyName System.Speech
$speaker=New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $speaker.SelectVoice('Microsoft Zira Desktop')
  $speaker.Rate=0
  $speaker.Volume=100
  for($index=0;$index -lt $data.scenes.Count;$index++){
    $destination=Join-Path $folder ('narration-'+$index+'.wav')
    $speaker.SetOutputToWaveFile($destination)
    $speaker.Speak($data.scenes[$index].narration)
    $speaker.SetOutputToNull()
    Write-Output $destination
  }
} finally {
  $speaker.Dispose()
}
