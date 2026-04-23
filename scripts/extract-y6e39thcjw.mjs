#!/usr/bin/env zx

class FixedQueue {

    constructor(){
        this.store = [];
        this.count = {
            true: 0,
            false: 0
        }
    }

    pop(){
        return this.store.pop();
    }

    push(target){
        console.log( target )
        process.exit(1)
        this.store.push(target)
        this.count[target] ++;
        if( this.store.length > 16 )
            this.store.pop();
    }
}

const files =
    await $`ls ./logs/*.jsonl`
        .quiet()
        .then(({stdout}) => stdout.split(/\n/g).filter(Boolean))
for( const file of files ){
    const { stdout } = await $`cat ${file}`.quiet()
    let count = -1;
    let last = {};
    let hist = new FixedQueue;
    for (const line of stdout.split(/\n/g).filter(Boolean)) {
        if( count == 395 )
            break;
        const parsed = JSON.parse(line);
        const { num_chunks, num_incorrect_chunks, num_correct_chunks } = parsed;
        let { last_num_chunks, last_num_correct_chunks, last_num_incorrect_chunks } = last;
        console.log( "last:", last )
        last = { num_chunks, num_incorrect_chunks, num_correct_chunks };
        console.log( "now:", last )
        if( count == -1 ){
            count ++;
            continue;
        }
        count ++;
        if( + num_incorrect_chunks > + last_num_incorrect_chunks ){
            console.log( true );
            hist.push(false);
        }
        else if( + num_correct_chunks > + last_num_correct_chunks ){
            console.log( false );
            hist.push(true);
        }
        else {
            console.log( "old" );
            let old = hist.pop();
            hist.push(old);
        };
    }
    console.log( file, count, hist )
}