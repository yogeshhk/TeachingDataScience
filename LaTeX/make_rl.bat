@echo off
for /r %%i in (Main*Reinforcement*.tex) do texify -cp %%i
