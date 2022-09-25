@echo off
for /r %%i in (TBD*.tex) do texify -cp %%i
