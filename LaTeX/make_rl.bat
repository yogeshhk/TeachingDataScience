@echo off
for /r %%i in (Main_*_Reinforcement*.tex) do texify -cp %%i
