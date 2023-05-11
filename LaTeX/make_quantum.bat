@echo off
for /r %%i in (Main_*Quantum*.tex) do texify -cp %%i
