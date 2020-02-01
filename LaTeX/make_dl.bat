@echo off
for /r %%i in (Main_Workshop_DeepLearning*.tex) do texify -cp %%i
