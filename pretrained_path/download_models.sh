#### PRIMEROOO
## install this --> git-lfs; on mac brew install git-lfs

# wav2vec2-base
git clone https://huggingface.co/facebook/wav2vec2-base/ wav2vec2-base/
git -C wav2vec2-base lfs pull
#rm -rf wav2vec2-base/.git/
#rm -f  wav2vec2-base/.gitattributes


git clone https://huggingface.co/facebook/wav2vec2-base-es-voxpopuli-v2 wav2vec2-base-es-voxpopuli-v2/
git -C wav2vec2-base-es-voxpopuli-v2 lfs pull
#rm -rf wav2vec2-base-es-voxpopuli-v2/.git/
#rm -f  wav2vec2-base-es-voxpopuli-v2/.gitattributes

echo "Fin."
