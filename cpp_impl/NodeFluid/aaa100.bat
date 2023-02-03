ffmpeg -framerate 100 -i "images/%%04d.png" -c:v libx264 -crf 1 -r 100 -pix_fmt yuv420p out.avi
pause