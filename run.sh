
def zero()
{
    for model in {'dep','tem'}
    do
    for sim in {'cos','dot'}
    do
    CUDA_VISIBLE_DEVICES=$1 python train.py  --sim $sim --offset 10 --train zero --model $model --batch_size 8
    done
    done

}
def zero_o()
{
    CUDA_VISIBLE_DEVICES=$1 python train.py  --sim cos --offset 10 --train zero --model dep --batch_size 8 &
    CUDA_VISIBLE_DEVICES=$2 python train.py  --sim cos --offset 10 --train zero --model tem --batch_size 8

}
def few_train()
{
    CUDA_VISIBLE_DEVICES=$1 python train.py  --sim cos --offset 10 --train few --model $2 --batch_size 8 --k $3
    CUDA_VISIBLE_DEVICES=$1 python train.py  --sim cos --offset 10 --train few --model $2 --batch_size 8 --k $4
    
}

def few()
{
    CUDA_VISIBLE_DEVICES=$1 python single_train.py  --sim cos --offset 10 --train few --model $2 --batch_size 8 --k $3
    CUDA_VISIBLE_DEVICES=$1 python single_train.py  --sim cos --offset 10 --train few --model $2 --batch_size 8 --k $4
    
}

def test()
{
    CUDA_VISIBLE_DEVICES=$1 python train.py  --sim cos --offset 10 --train zero --model $2 --test 0
    CUDA_VISIBLE_DEVICES=$1 python train.py  --sim cos --offset 10 --train zero --model $2 --test 1
    CUDA_VISIBLE_DEVICES=$1 python train.py  --sim cos --offset 10 --train zero --model $2 --test 12
    CUDA_VISIBLE_DEVICES=$1 python train.py  --sim cos --offset 10 --train zero --model $2 --test 13
}

CUDA_VISIBLE_DEVICES=0 python train.py  --sim cos --offset 10 --train few --model dep --batch_size 8 --k 5&
CUDA_VISIBLE_DEVICES=1 python train.py  --sim cos --offset 10 --train few --model tem --batch_size 8 --k 5&
CUDA_VISIBLE_DEVICES=2 python single_train.py  --sim cos --offset 10 --train few --model dep --batch_size 8 --k 5&
CUDA_VISIBLE_DEVICES=3 python single_train.py  --sim cos --offset 10 --train few --model tem --batch_size 8 --k 5&
CUDA_VISIBLE_DEVICES=7 python single_train.py  --sim cos --offset 10 --train few --model cls --batch_size 8 --k 5&
CUDA_VISIBLE_DEVICES=5 python single_train.py  --sim cos --offset 10 --train few --model cls_dep --batch_size 8 --k 5
