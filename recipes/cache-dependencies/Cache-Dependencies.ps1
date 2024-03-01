$path = '.\dependencies'

if (Test-Path $path) {
    Remove-Item -Force -Recurse $path
}

New-Item -ItemType Directory -Path $path -Force

& pip download  -r requirements.txt -d $path