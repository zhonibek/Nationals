param([int]$Port = 8080, [int]$Threads = 4, [int]$CacheMiB = 128)
$ErrorActionPreference = 'Stop'
throw 'The legacy CPU launcher is disabled. Nemotron must run on an explicitly verified NVIDIA T4 with full GPU offload. Configure an authenticated connection to that runtime; no T4 runtime is started or verified by this script.'
