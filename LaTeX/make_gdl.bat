@echo off
for /r %%i in (Main_*Geometric*.tex) do texify -cp %%i
