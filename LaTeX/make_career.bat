@echo off
for /r %%i in (Main_*Career*.tex) do texify -cp %%i
