@echo off
for /r %%i in (Main*.tex) do texify -cp %%i
