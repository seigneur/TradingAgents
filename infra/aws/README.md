# AWS setup for signal tracking

The daily workflow saves each ticker's latest prediction to the DynamoDB table
`tradingagents-signals` (us-east-1, 1 RCU / 1 WCU provisioned: inside the always-free tier).
`scripts/signal_watch.py` runs hourly on the local box and alerts Telegram when price moves
against the latest prediction.

GitHub Actions logs in to AWS through OIDC, so no AWS keys are stored in GitHub. The role can
only `PutItem` on that one table, and only from `main` of `seigneur/TradingAgents`. The repo uses GitHub's immutable OIDC subject
(`repo:seigneur@1625844/TradingAgents@1397720142:...`), so the trust policy must use that form.

## One-time setup (run from the repo root)

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com --client-id-list sts.amazonaws.com
aws iam create-role --role-name tradingagents-gha-signals \
  --assume-role-policy-document file://infra/aws/github-trust.json
aws iam put-role-policy --role-name tradingagents-gha-signals \
  --policy-name signals-write --policy-document file://infra/aws/signals-write.json
gh variable set AWS_ROLE_ARN \
  --body arn:aws:iam::365600189165:role/tradingagents-gha-signals
```

Until `AWS_ROLE_ARN` is set, the workflow skips the AWS step and runs as before.
