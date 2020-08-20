@echo off
for /r %%i in (Main_*_DeepLearning*.tex) do texify -cp %%i
