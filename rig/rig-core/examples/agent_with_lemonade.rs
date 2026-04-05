/// Connection test for the [Lemonade](https://github.com/lemonade-sdk/lemonade) local AI server.
///
/// Requires a running Lemonade server (default: http://localhost:13305).
/// Set LEMONADE_SERVER_URL to override. Set LEMONADE_MODEL to choose a model
/// (run `lemonade-server list` to see available model names).
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::lemonade;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = lemonade::Client::from_env();

    let model = std::env::var("LEMONADE_MODEL").expect("LEMONADE_MODEL not set");

    let agent = client
        .agent(model)
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent.prompt("Say hello in one sentence.").await?;
    println!("{response}");

    Ok(())
}
