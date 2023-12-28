mode=$1

if [$mode -eq $(0)];
then
  python demo/demo.py webcam --config config/nanodet-plus-m-1.5x_416.yml --model model/nanodet-plus-m-1.5x_416.pth --camid 0
else
  echo "Unexisting mode"
fi