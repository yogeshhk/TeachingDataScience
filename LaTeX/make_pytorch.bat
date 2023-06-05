@echo off
for /r %%i in (Main_*Pytorch_*.tex) do texify -cp %%i
