mod aim;
mod ballistics;
mod map;
mod math;
mod nav;
mod protocol;
mod tactics;
mod world;

use protocol::{Message, Obs};
use std::io::{self, BufRead, BufWriter, StdinLock, Write};
use std::sync::{Arc, Condvar, Mutex};
use tactics::Bot;

#[derive(Default)]
struct Inbox {
    latest: Option<Box<Obs>>,
    ended: bool,
}

fn read_latest(mut input: StdinLock<'_>, inbox: &Arc<(Mutex<Inbox>, Condvar)>) {
    let mut line = String::new();
    loop {
        line.clear();
        match input.read_line(&mut line) {
            Ok(n) if n > 0 => {}
            _ => break,
        }
        match serde_json::from_str::<Message>(&line) {
            Ok(Message::Obs(obs)) => {
                let mut state = inbox.0.lock().expect("inbox lock");
                if state
                    .latest
                    .as_ref()
                    .is_none_or(|next| obs.tick > next.tick)
                {
                    state.latest = Some(obs);
                }
                inbox.1.notify_one();
            }
            Ok(Message::End) => break,
            _ => {}
        }
    }
    inbox.0.lock().expect("inbox lock").ended = true;
    inbox.1.notify_one();
}

fn send(output: &mut impl Write, message: &impl serde::Serialize) -> io::Result<()> {
    serde_json::to_writer(&mut *output, message)?;
    output.write_all(b"\n")?;
    output.flush()
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = io::stdin().lock();
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        return Ok(());
    }
    let Message::Hello(hello) = serde_json::from_str(&line)? else {
        return Err("expected hello".into());
    };
    if hello.protocol != "mechfight/1" {
        return Err(format!("unsupported protocol {}", hello.protocol).into());
    }
    drop(input);

    let mut bot = Bot::new(hello)?;
    let mut output = BufWriter::new(io::stdout().lock());
    send(
        &mut output,
        &serde_json::json!({"type":"ready","name":"Mechtoyale V2"}),
    )?;
    let inbox = Arc::new((Mutex::new(Inbox::default()), Condvar::new()));
    let reader_inbox = Arc::clone(&inbox);
    std::thread::Builder::new()
        .name("stdin-reader".into())
        .spawn(move || read_latest(io::stdin().lock(), &reader_inbox))?;
    let mut last_tick = None;
    loop {
        let mut state = inbox.0.lock().expect("inbox lock");
        while state.latest.is_none() && !state.ended {
            state = inbox.1.wait(state).expect("inbox lock");
        }
        let Some(obs) = state.latest.take() else {
            return Ok(());
        };
        drop(state);
        if last_tick.is_some_and(|last| last >= obs.tick) {
            continue;
        }
        last_tick = Some(obs.tick);
        send(&mut output, &bot.decide(&obs))?;
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mechbot: {error}");
        std::process::exit(1);
    }
}
