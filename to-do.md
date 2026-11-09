# Description
Small ML model plays go against itself perpetually. Each training iteration loads the current model, plays 100 self-play games, rewards wins with +1 and losses with -1, then updates the neural network. Front end web page allows you to see the end result of the latest game and replay it step by step. GitHub Actions runs one training iteration daily.

- [x] implement 9x9 go board 
- [x] implement placing stones
- [x] implement taking turns
- [x] implement rules
- [x] implement simple ML model for playing go
- [x] implement script to train the model one iteration
- [x] update script to update front end (show end state and allow to replay the game turn by turn)
- [x] simple css
- [x] github actions script to automate model training and front end updating
