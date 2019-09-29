@echo off
for /r %%i in (Main_*_Midcurve*.tex) do texify -cp %%i
