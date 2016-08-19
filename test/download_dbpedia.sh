#!/usr/bin/env bash
# Modified version of fasttext/cpp/classification-example.sh

# Download and normalize data dbpedia.train
# Run:
# % sh test/download_dbpedia.sh

myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

normalize_text() {
  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " " | myshuf
}

echo "Downloading the dbpedia_csv.tar.gz ..."
wget -c "https://googledrive.com/host/0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k" \
    -O test/dbpedia_csv.tar.gz

echo "Extract dbpedia_csv.tar.gz to test/"
tar xzvf test/dbpedia_csv.tar.gz -C test/

echo "Creating the test/dbpedia.train ..."
cat test/dbpedia_csv/train.csv | normalize_text > test/dbpedia.train
