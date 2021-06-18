# DiffPool
python -m train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --num-classes=6 --method=soft-assign --goren_loss_type=None --optim='sgd' --temp=False

# DiffPool - softmax with temperature
python -m train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --num-classes=6 --method=soft-assign --goren_loss_type=None --optim='sgd'
 
# additional loss term: combinatorial Laplacians & quadratic form
python -m train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --num-classes=6 --method=soft-assign --goren_loss_type=0 --optim='sgd'
 
# additional loss term: combinatorial Laplacian for the original graph and doubly-weighted Laplacian for the coarse graph & Rayleigh quotient
python -m train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --num-classes=6 --method=soft-assign --goren_loss_type=1 --optim='sgd'

# use accuracy.py after traning
# json files with results are also provided