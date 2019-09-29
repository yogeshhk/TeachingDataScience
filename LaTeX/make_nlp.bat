@echo off
for /r %%i in (Main_Workshop_Natural*.tex) do texify -cp %%i
