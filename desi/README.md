# DESI

In this directory are stored the config files used to run on DESI data / mocks.
Typically, after installing *cosmodesi*, running likelihood profiling / sampling and producing debugging plots given a configuration file config.yaml is simply:
```
cosmofit profile config.yaml  # or cosmofit sample config.yaml
cosmofit summarize config.yaml  # triangle plots, marginalized errors, etc.
cosmofit do config.yaml  # any custom thing, e.g. plot of best fit power spectrum model
```
