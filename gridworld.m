env = rlPredefinedEnv("BasicGridWorld")
env.ResetFcn = @() 5;
plot(env)