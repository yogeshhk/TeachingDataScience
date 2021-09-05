@echo off
for /r %%i in (Main_Workshop_Quantum*.tex) do texify -cp %%i
