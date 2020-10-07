@echo off
for /r %%i in (Main_*Swift4TensorFlow*.tex) do texify -cp %%i
