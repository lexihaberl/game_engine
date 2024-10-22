cd
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup component add rust-analyzer # maybe have to resource .bashrc beforehand
git clone repo
#sudo apt-get install build-essential git
sudo dnf install make automake gcc gcc-c++ cmake git libxkbcommon


## vulkan
#sudo apt-get install libvulkan-dev vulkan-tools glslc
sudo dnf install vulkan-devel vulkan-tools glslc  vulkan-validation-layers    


## nvim
LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po '"tag_name": "v\K[^"]*')
curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz"
tar xf lazygit.tar.gz lazygit
sudo install lazygit /usr/local/bin
#sudo apt-get install ripgrep fd-find
sudo dnf install ripgrep fd-find
curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz
sudo rm -rf /opt/nvim
sudo tar -C /opt -xzf nvim-linux64.tar.gz
echo 'export PATH="$PATH:/opt/nvim-linux64/bin"' >> ~/.bashrc
sudo dnf install nodejs
git clone https://github.com/LazyVim/starter ~/.config/nvim
rm -rf ~/.config/nvim/.git
rustup component add rust-analyzer
