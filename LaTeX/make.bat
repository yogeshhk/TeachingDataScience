@echo off
for /r %%i in (Main*QuantumComputing*.tex) do texify -cp %%i
