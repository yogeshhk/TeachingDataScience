@echo off
for /r %%i in (Main_Seminar_*Sarvam*.tex) do texify --engine=luatex  -cp %%i
