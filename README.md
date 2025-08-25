# Facial-expression-conversion
本プログラムは、入力顔画像と目標 Action Units（AU）を条件に、人物の同一性を保ったまま表情のみを変換します。Stable Diffusionを使用し、q/k/v への LoRA で軽量学習。AffectNet で前処理（顔切り出し・整列）を行い、AU は正規化して強度を連続制御可能。学習は自己再構成＋識別保持を両立し、推論は img2img で微細編集。評価は AU MSE・ArcFace ・LPIPS を用いて自然さと一貫性を確認します。
