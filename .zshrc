########## Custom settings ##########
source /opt/intel/oneapi/setvars.sh --force
for f in $(find /opt/intel/oneapi -name \*.cmake | grep -v /doc/ | grep -v /examples/ | xargs -L 1 dirname | sort | uniq); do
	CMAKE_PREFIX_PATH=$f:${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}
done
export CMAKE_PREFIX_PATH
#################### git ####################
autoload -Uz vcs_info
zstyle ':vcs_info:git:*' check-for-changes true
zstyle ':vcs_info:git:*' stagedstr "%F{cyan}"
zstyle ':vcs_info:git:*' unstagedstr "%F{magenta}"
zstyle ':vcs_info:git:*' formats '%u%c(%r)-[%b]%f' '%c%u %m'
zstyle ':vcs_info:git:*' actionformats '%u%c(%r)-[%b|%a]%f' '%c%u %m'

################### prompts ####################
setopt prompt_subst
Date=$(date "+%Y/%m/%d")
precmd () {
   PROMPT="%(?.%F{cyan}.%F{red})$Date %* %d%f"
   PROMPT+="
%(?..%F{red})%n@%m %(?..(%?%)%f)%# "
}
precmd_vcs_info() { vcs_info }
precmd_functions+=( precmd_vcs_info )
RPROMPT=\$vcs_info_msg_0_

#################### auto-completion ####################
autoload -Uz compinit && compinit
zstyle ':completion:*' menu select

#################### alias ####################
alias ls='ls --color=auto'
alias grep='grep --color=auto'
alias crontab='crontab -i'

#################### history ####################
export HISTFILE=${HOME}/.zhistory
export HISTSIZE=1000
export SAVEHIST=100000
export IGNOREEOF=1000

setopt EXTENDED_HISTORY
setopt hist_ignore_dups
setopt hist_ignore_all_dups
setopt hist_save_no_dups
setopt hist_ignore_space
setopt hist_verify
setopt hist_reduce_blanks
setopt hist_expand
setopt inc_append_history

autoload history-search-end
zle -N history-beginning-search-backward-end history-search-end
zle -N history-beginning-search-forward-end history-search-end
bindkey "^R" history-incremental-search-backward
bindkey "^S" history-incremental-search-forward
bindkey "^p" history-beginning-search-backward-end
bindkey "^n" history-beginning-search-forward-end
bindkey "^u" backward-kill-line
bindkey "^w" kill-region