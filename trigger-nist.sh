DIR=nist/data/lucene-index-cord19-abstract-2020-05-01
TAR=lucene-index-cord19-abstract-2020-05-01.tar.gz

if [ ! -d "$DIR" ]; then
    wget -nc https://www.dropbox.com/s/wxjoe4g71zt5za2/${TAR}
    tar -xvzf ${TAR} -C nist/data
fi

python nist/nist.py --k 0 &
python nist/nist.py --k 1000
python nist/nist.py --k 2500 &
python nist/nist.py --k 5000
