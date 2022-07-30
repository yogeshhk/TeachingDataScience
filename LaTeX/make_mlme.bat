@echo off
for /r %%i in (Main_*MechEngg*.tex) do texify -cp %%i
